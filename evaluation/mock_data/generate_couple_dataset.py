"""情侣对话数据集生成器

用于测试 AI 女友记忆系统：构造多轮对话，让男主在对话中自然透露 profile 信息，
最终产出 (1) 多轮对话、(2) profile fact 列表、(3) 测试 query 三类数据。

工作流（三步走，确保 fact 与对话精确对齐）：
    1) 生成男主人设 + AI 女友人设 + 完整 profile_facts 计划表（每个 fact 预先分配 planned_session_id）
    2) 逐 session 生成对话；同时让 LLM 回填每个 fact 实际对应的 dia_id
    3) 基于完整数据生成混合难度 QA

输出格式扩展自 locomo10.json，增加 profile_facts 字段。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


DEFAULT_ENV = r"D:\aiworks\code\memS\.env"
DEFAULT_OUTPUT_DIR = r"D:\aiworks\code\memS\evaluation\mock_data\couple_dataset"
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"

FACT_CATEGORIES = [
    "basic",         # 姓名、年龄、生日、家乡、星座等
    "work",          # 职业、公司、专业、学历、工作年限
    "preference",    # 食物、音乐、电影、颜色、宠物等爱好
    "habit",         # 作息、运动、饮食禁忌、过敏
    "relationship",  # 家人、朋友、宠物名字
    "history",       # 毕业学校、前任、童年故事
    "plan",          # 短期长期目标
    "emotion",       # 情感状态、性格特点
]

QA_CATEGORIES = {
    "single_fact": "针对单个 profile fact 的直接提问",
    "multi_fact": "需要综合 2-3 个 fact 才能回答",
    "adversarial": "反事实/否定问题（事实未提及或与事实相反）",
    "temporal": "涉及时间推理（例如某事发生在哪个时间段）",
}


# ---------- 工具函数 ----------


def load_env_like(path: str) -> tuple[str, str, str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f".env not found: {path}")
    key = ""
    model = ""
    base_url = DEFAULT_BASE_URL
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k_upper = k.strip().upper()
            v = v.strip().strip("\"' ")
            if k_upper in {"DEEPSEEK_API_KEY", "OPENAI_API_KEY", "API_KEY"} and v:
                key = v
            elif k_upper in {"DEEPSEEK_MODEL", "MODEL"} and v:
                model = v
            elif k_upper in {"DEEPSEEK_BASE_URL", "BASE_URL"} and v:
                base_url = v
    if not key:
        raise ValueError("API key not found in env file.")
    if not model:
        model = "deepseek-chat"
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    if not normalized.endswith("/v1") and not normalized.endswith(".com"):
        normalized = normalized + "/v1"
    return key, model, normalized


def extract_json(content: str) -> Any:
    """从 LLM 输出中抽取最外层 JSON 对象/数组。容忍 markdown code fence。"""
    content = content.strip()
    # 优先剥离 markdown 代码块
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    # 直接尝试
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    # 退而求其次：找第一个完整对象 / 数组
    for opener, closer in [("{", "}"), ("[", "]")]:
        start = content.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(content)):
            ch = content[i]
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    snippet = content[start : i + 1]
                    try:
                        return json.loads(snippet)
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"无法从模型输出中解析 JSON。原始内容前 500 字：\n{content[:500]}")


def call_llm(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    *,
    temperature: float = 0.7,
    max_retries: int = 5,
    label: str = "",
) -> str:
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                stream=False,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            last_err = e
            sleep_s = min(2 ** attempt, 30)
            print(f"  [{label}] attempt {attempt} failed: {e}; retry in {sleep_s}s", flush=True)
            time.sleep(sleep_s)
    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_err}")


# ---------- 核心生成步骤 ----------


@dataclass
class SampleConfig:
    sample_id: str
    num_sessions: int = 8
    turns_per_session_min: int = 12
    turns_per_session_max: int = 18
    facts_per_category_target: int = 4  # 每类 fact 平均期望数量
    seed: int = 0


def build_persona_and_plan(
    client: OpenAI,
    model: str,
    cfg: SampleConfig,
) -> dict[str, Any]:
    """Step 1: 生成男主/AI 女友人设 + 完整 profile_facts 计划表 + 每个 session 主题与时间。"""

    facts_total = cfg.facts_per_category_target * len(FACT_CATEGORIES)

    system = (
        "你是中文对话数据集的设定师。你需要为「AI 女友记忆系统」评测构造一个独立样本。"
        "请严格按要求输出 JSON，不要解释，不要写多余前后缀。"
    )

    user = f"""请生成一份情侣关系数据集的「样本设定」，包含以下三个部分：

1. speaker_a_persona：男主人设（包含 name、age、basic_summary 简要介绍 50-100 字）
2. speaker_b_persona：AI 女友人设（包含 name、personality_summary 性格特点 50-100 字，体现温柔、记性好、善于倾听）
3. profile_facts：男主的 profile fact 列表，共 {facts_total} 条左右；
   - 每条 fact 包含字段：fact_id（如 F1, F2 …）、category（取值范围：{FACT_CATEGORIES}）、content（一句话陈述事实，主语统一为男主名字）、
     planned_session_id（取值 session_1 ~ session_{cfg.num_sessions}）、naturalness_hint（一句话提示如何在对话中自然引出，便于后续生成对话）
   - 每个 category 至少 {cfg.facts_per_category_target - 1} 条；
   - planned_session_id 应均匀分布在 session_1 到 session_{cfg.num_sessions}，且尽量让基本信息在前面 session 出现、长期目标/情感等放在后面 session；
   - fact 之间互不矛盾，且要有真实感（避免老套刻板印象）。
4. session_plan：一个长度为 {cfg.num_sessions} 的数组，每个元素描述一个 session：
   - session_id（session_1, session_2, …）
   - date_time（中文风格的具体时间，例如 "2024年3月15日 晚上21:30"，整体跨度 3-6 个月，session 之间间隔从几天到几周不等，时间需要单调递增）
   - topic（一句话场景，例如 "周末小情侣的居家闲聊"、"男主刚下班疲惫诉苦"、"深夜失眠互聊童年"）
   - emotion_tone（例如 "轻松甜蜜"、"安静温柔"、"略带焦虑"）
   - target_fact_ids（数组，列出所有 planned_session_id == 当前 session 的 fact_id；必须与 profile_facts 完全一致）

输出 JSON 结构示例：
{{
  "speaker_a_persona": {{ "name": "...", "age": 28, "basic_summary": "..." }},
  "speaker_b_persona": {{ "name": "...", "personality_summary": "..." }},
  "profile_facts": [ {{ "fact_id": "F1", "category": "basic", "content": "...", "planned_session_id": "session_1", "naturalness_hint": "..." }} ],
  "session_plan": [ {{ "session_id": "session_1", "date_time": "...", "topic": "...", "emotion_tone": "...", "target_fact_ids": ["F1", "F3"] }} ]
}}

请直接输出 JSON。男主的名字、AI 女友的名字、年龄、职业等都自由发挥但要符合中国年轻情侣的真实感。
随机种子：{cfg.seed}（仅作为风格扰动提示）。"""

    raw = call_llm(client, model, system, user, temperature=0.9, label=f"{cfg.sample_id}/persona")
    plan = extract_json(raw)

    # 基本字段校验
    for key in ["speaker_a_persona", "speaker_b_persona", "profile_facts", "session_plan"]:
        if key not in plan:
            raise ValueError(f"persona 输出缺失字段：{key}")
    if len(plan["session_plan"]) != cfg.num_sessions:
        raise ValueError(
            f"session_plan 数量 {len(plan['session_plan'])} != 期望 {cfg.num_sessions}"
        )

    # 确保 target_fact_ids 与 planned_session_id 一致
    facts_by_session: dict[str, list[str]] = {}
    for fact in plan["profile_facts"]:
        facts_by_session.setdefault(fact["planned_session_id"], []).append(fact["fact_id"])
    for sess in plan["session_plan"]:
        sess["target_fact_ids"] = facts_by_session.get(sess["session_id"], [])

    return plan


def build_session_dialog(
    client: OpenAI,
    model: str,
    cfg: SampleConfig,
    plan: dict[str, Any],
    session_idx: int,
    previous_summaries: list[str],
) -> dict[str, Any]:
    """Step 2: 生成单个 session 的对话，并回填 fact 实际出现的 dia_id。"""

    session_meta = plan["session_plan"][session_idx]
    session_id = session_meta["session_id"]  # session_N
    session_num = session_idx + 1

    target_fact_ids = session_meta.get("target_fact_ids", [])
    target_facts = [
        f for f in plan["profile_facts"] if f["fact_id"] in target_fact_ids
    ]

    speaker_a = plan["speaker_a_persona"]["name"]
    speaker_b = plan["speaker_b_persona"]["name"]

    prev_block = "\n".join(f"- {s}" for s in previous_summaries) if previous_summaries else "（这是第一段对话）"

    target_facts_block = json.dumps(
        [
            {
                "fact_id": f["fact_id"],
                "category": f["category"],
                "content": f["content"],
                "naturalness_hint": f.get("naturalness_hint", ""),
            }
            for f in target_facts
        ],
        ensure_ascii=False,
        indent=2,
    )

    system = (
        "你是中文情侣日常聊天对话生成器，擅长写自然、口语化、有生活气息的微信式对话。"
        "请严格按要求输出 JSON，不要解释。"
    )

    user = f"""请为情侣记忆数据集生成第 {session_num} 段对话（session_id = {session_id}）。

【人物】
- 男主（speaker_a）：{speaker_a}，{plan['speaker_a_persona'].get('basic_summary', '')}
- AI 女友（speaker_b）：{speaker_b}，{plan['speaker_b_persona'].get('personality_summary', '')}

【时间】{session_meta['date_time']}
【场景主题】{session_meta['topic']}
【情绪基调】{session_meta['emotion_tone']}

【之前对话回顾（仅供保持设定一致，不要重复）】
{prev_block}

【本场必须自然引出的 fact 列表】（来源于男主之口，最好通过情境化叙述而非干巴巴罗列）
{target_facts_block}

【生成要求】
1. 对话总轮数 {cfg.turns_per_session_min}-{cfg.turns_per_session_max} 轮；男女发言交替为主，可有连续两条但不要超过两条；
2. 男主的发言中必须自然包含上面所有 fact 的信息（每条 fact 至少在某一句中体现，不要生硬列举）；
3. 语气要符合 21-32 岁中国情侣日常聊天，口语化、可有少量 emoji（适度，不要泛滥），可有打趣、撒娇、关心；
4. AI 女友（{speaker_b}）的回复要体现倾听、共情、记忆能力（偶尔提到之前 session 的细节也可以，但不要造假新事实）；
5. 每条对话格式：{{ "speaker": "{speaker_a} 或 {speaker_b}（必须填人名，不要写 speaker_a/speaker_b）", "dia_id": "D{session_num}:N", "text": "..." }}，dia_id 序号从 1 开始连续；
6. 输出 JSON 字段：
   - "session_dialog": 上述对话数组
   - "fact_evidence": 数组，每个元素 {{ "fact_id": "F?", "dia_ids": ["D{session_num}:?"] }}，标注每个目标 fact 实际出现在哪些 dia_id（同一 fact 可对应多条 dia_id；至少 1 条；只列男主说的）
   - "session_summary": 一句话总结本场对话（≤60 字，便于下一 session 参考）

请直接输出 JSON。"""

    raw = call_llm(
        client,
        model,
        system,
        user,
        temperature=0.85,
        label=f"{cfg.sample_id}/{session_id}",
    )
    parsed = extract_json(raw)

    for key in ["session_dialog", "fact_evidence", "session_summary"]:
        if key not in parsed:
            raise ValueError(f"{session_id} 输出缺失字段：{key}")

    speaker_a = plan["speaker_a_persona"]["name"]
    speaker_b = plan["speaker_b_persona"]["name"]

    # 校正 dia_id 前缀 + 规范化 speaker 字段（统一为人物姓名，避免模型写成 speaker_a/speaker_b）
    for i, turn in enumerate(parsed["session_dialog"], start=1):
        turn["dia_id"] = f"D{session_num}:{i}"
        if "speaker" not in turn or "text" not in turn:
            raise ValueError(f"{session_id} 第 {i} 条对话字段缺失")
        spk = str(turn["speaker"]).strip()
        spk_norm = spk.lower().replace(" ", "").replace("_", "")
        if spk_norm in {"speakera", "a", "male", "boy", "boyfriend"}:
            turn["speaker"] = speaker_a
        elif spk_norm in {"speakerb", "b", "female", "girl", "girlfriend", "ai"}:
            turn["speaker"] = speaker_b
        elif spk == speaker_a or spk == speaker_b:
            pass
        else:
            print(
                f"  [warn] {session_id} D{session_num}:{i} speaker='{spk}' 无法识别，按内容启发式判断",
                flush=True,
            )
            # 简单启发：如果上一条是 a，本条认为是 b，反之亦然
            if i > 1:
                prev = parsed["session_dialog"][i - 2]["speaker"]
                turn["speaker"] = speaker_b if prev == speaker_a else speaker_a
            else:
                turn["speaker"] = speaker_b  # AI 女友通常先打招呼

    # 检查 fact_evidence 完整性
    covered = {fe["fact_id"] for fe in parsed["fact_evidence"]}
    missing = [fid for fid in target_fact_ids if fid not in covered]
    if missing:
        print(
            f"  [warn] {session_id} fact 未覆盖：{missing}，将由后续 QA 阶段视为弱证据",
            flush=True,
        )

    return parsed


def build_qa(
    client: OpenAI,
    model: str,
    cfg: SampleConfig,
    plan: dict[str, Any],
    full_conversation: dict[str, Any],
    profile_facts_with_evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Step 3: 基于完整数据生成混合难度 QA。"""

    speaker_a = plan["speaker_a_persona"]["name"]
    speaker_b = plan["speaker_b_persona"]["name"]

    facts_block = json.dumps(
        [
            {
                "fact_id": f["fact_id"],
                "category": f["category"],
                "content": f["content"],
                "source_session_ids": f["source_session_ids"],
                "evidence": f["evidence"],
            }
            for f in profile_facts_with_evidence
        ],
        ensure_ascii=False,
        indent=2,
    )

    target_total = max(20, len(profile_facts_with_evidence))

    system = (
        "你是中文记忆系统评测题目设计师。请基于给定的 profile facts 设计高质量评测问题。"
        "题目要求自然、口语化、避免简单复制原句，输出严格 JSON。"
    )

    user = f"""【背景】下面是「{speaker_a}」（男主）和「{speaker_b}」（AI 女友）的对话所衍生出的 profile_facts：

{facts_block}

【任务】请生成约 {target_total} 道测试题，难度分布大致为：
- single_fact（约 50%）：直接询问单个 fact，例如「我多大了？」「我家乡在哪？」
- multi_fact（约 25%）：需要综合 2-3 个 fact 才能回答，例如「根据我说过的，我下班后通常做什么？」
- adversarial（约 15%）：反事实/否定/陷阱题，正确答案应当指出"未提及"或与提问中的错误前提矛盾
- temporal（约 10%）：与时间相关的推理题（例如"我是哪个 session 提到换工作的？" 或 "我今年几岁？" 隐含时间）

每道题输出字段：
- question_id（Q1, Q2, …）
- question（用第一人称「我」提问，假设男主在向 AI 女友测试记忆；中文，自然口语）
- answer（简短中文标准答案；adversarial 题答案可以是「未提及」「未告知」等）
- evidence_facts（数组，列出回答需要的 fact_id；adversarial 题可为空数组）
- evidence（数组，列出关键 dia_id；adversarial 题可为空数组）
- category（single_fact / multi_fact / adversarial / temporal）
- difficulty（easy / medium / hard）

【硬性要求】
1. 至少覆盖到 80% 的 fact_id；
2. 题目不要重复同一信息点，问法要多样化（用、是、什么时候、哪里、为什么、有没有 等）；
3. adversarial 题中的"错误前提"要合理可信（不能瞎编与事实毫不沾边的内容）；
4. 不要在题面里直接抄 fact 原句；
5. 男主的名字「{speaker_a}」可以出现在题面中作为代词替代。

请直接输出 JSON：{{ "qa": [ ... ] }}。"""

    raw = call_llm(client, model, system, user, temperature=0.7, label=f"{cfg.sample_id}/qa")
    parsed = extract_json(raw)
    if "qa" not in parsed:
        raise ValueError("QA 输出缺失 qa 字段")
    return parsed["qa"]


# ---------- 组装 ----------


def assemble_profile_facts(
    plan: dict[str, Any],
    fact_evidence_per_session: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """合并 plan 中的 facts 和每个 session 实际回填的 dia_id。"""
    evidence_lookup: dict[str, list[str]] = {}
    source_sessions: dict[str, set[str]] = {}
    for session_id, fe_list in fact_evidence_per_session.items():
        for fe in fe_list:
            fid = fe["fact_id"]
            dids = fe.get("dia_ids") or []
            evidence_lookup.setdefault(fid, []).extend(dids)
            if dids:
                source_sessions.setdefault(fid, set()).add(session_id)

    enriched: list[dict[str, Any]] = []
    for f in plan["profile_facts"]:
        fid = f["fact_id"]
        sessions = sorted(source_sessions.get(fid, {f["planned_session_id"]}))
        enriched.append(
            {
                "fact_id": fid,
                "category": f["category"],
                "content": f["content"],
                "source_session_ids": sessions,
                "evidence": list(dict.fromkeys(evidence_lookup.get(fid, []))),
            }
        )
    return enriched


def build_one_sample(
    client: OpenAI,
    model: str,
    cfg: SampleConfig,
) -> dict[str, Any]:
    print(f"=== building sample: {cfg.sample_id} ===", flush=True)
    print("[1/3] 生成人设与 fact 计划表 ...", flush=True)
    plan = build_persona_and_plan(client, model, cfg)

    speaker_a = plan["speaker_a_persona"]["name"]
    speaker_b = plan["speaker_b_persona"]["name"]
    print(
        f"  人设：男主 {speaker_a} / AI 女友 {speaker_b}；总 fact {len(plan['profile_facts'])} 条",
        flush=True,
    )

    print("[2/3] 逐 session 生成对话 ...", flush=True)
    conversation: dict[str, Any] = {
        "speaker_a": speaker_a,
        "speaker_b": speaker_b,
    }
    fact_evidence_per_session: dict[str, list[dict[str, Any]]] = {}
    summaries: list[str] = []

    for idx, sess in enumerate(plan["session_plan"]):
        sess_id = sess["session_id"]
        print(f"  - {sess_id} ({sess['date_time']}): {sess['topic']}", flush=True)
        result = build_session_dialog(client, model, cfg, plan, idx, summaries)
        conversation[f"{sess_id}_date_time"] = sess["date_time"]
        conversation[sess_id] = result["session_dialog"]
        fact_evidence_per_session[sess_id] = result["fact_evidence"]
        summaries.append(f"[{sess_id} {sess['date_time']}] {result['session_summary']}")

    print("[3/3] 生成测试 QA ...", flush=True)
    profile_facts = assemble_profile_facts(plan, fact_evidence_per_session)
    qa = build_qa(client, model, cfg, plan, conversation, profile_facts)

    sample = {
        "sample_id": cfg.sample_id,
        "speaker_a_persona": plan["speaker_a_persona"],
        "speaker_b_persona": plan["speaker_b_persona"],
        "session_plan": [
            {k: v for k, v in s.items() if k != "target_fact_ids"}
            | {"target_fact_ids": s.get("target_fact_ids", [])}
            for s in plan["session_plan"]
        ],
        "conversation": conversation,
        "profile_facts": profile_facts,
        "qa": qa,
    }
    return sample


# ---------- CLI ----------


def main() -> int:
    parser = argparse.ArgumentParser(description="情侣对话数据集生成器（用于 AI 女友记忆系统评测）")
    parser.add_argument("--num-samples", type=int, default=1, help="生成多少个独立样本")
    parser.add_argument("--num-sessions", type=int, default=8, help="每个样本的 session 数")
    parser.add_argument("--turns-min", type=int, default=12, help="每个 session 的最少对话轮数")
    parser.add_argument("--turns-max", type=int, default=18, help="每个 session 的最多对话轮数")
    parser.add_argument(
        "--facts-per-category",
        type=int,
        default=4,
        help="每类 fact 的目标数量（共 8 类）",
    )
    parser.add_argument("--start-index", type=int, default=1, help="样本编号起始")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（用于风格扰动）")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--env", type=str, default=DEFAULT_ENV)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="若输出文件已存在则跳过该样本（默认开启）",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="即使已存在也重新生成（覆盖）",
    )
    args = parser.parse_args()

    api_key, model, base_url = load_env_like(args.env)
    print(f"Model: {model}; Base URL: {base_url}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    client = OpenAI(api_key=api_key, base_url=base_url)

    rng = random.Random(args.seed)

    for i in range(args.num_samples):
        sample_idx = args.start_index + i
        sample_id = f"couple_{sample_idx:03d}"
        out_path = os.path.join(args.output_dir, f"{sample_id}.json")
        if args.skip_existing and os.path.exists(out_path):
            print(f"=== {sample_id} 已存在，跳过 ===", flush=True)
            rng.randint(0, 10**9)  # 仍消耗一次 rng，让后续 seed 一致
            continue
        cfg = SampleConfig(
            sample_id=sample_id,
            num_sessions=args.num_sessions,
            turns_per_session_min=args.turns_min,
            turns_per_session_max=args.turns_max,
            facts_per_category_target=args.facts_per_category,
            seed=rng.randint(0, 10**9),
        )
        try:
            sample = build_one_sample(client, model, cfg)
        except Exception as e:  # noqa: BLE001
            print(f"!!! sample {cfg.sample_id} 生成失败：{e}", flush=True)
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)
        print(f"  -> 已写入 {out_path}", flush=True)

    # 扫描目录下所有 couple_*.json 重建合并文件
    all_samples: list[dict[str, Any]] = []
    for name in sorted(os.listdir(args.output_dir)):
        if not (name.startswith("couple_") and name.endswith(".json")):
            continue
        p = os.path.join(args.output_dir, name)
        try:
            with open(p, "r", encoding="utf-8") as f:
                all_samples.append(json.load(f))
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] 跳过损坏文件 {name}: {e}", flush=True)

    if all_samples:
        merged_path = os.path.join(args.output_dir, "all_samples.json")
        with open(merged_path, "w", encoding="utf-8") as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=2)
        print(f"\n合并文件：{merged_path}（共 {len(all_samples)} 个样本）", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

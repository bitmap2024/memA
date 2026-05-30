#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统一记忆抽取器：一次 LLM 调用同时抽取 profile / episodic / state 三类记忆。"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from loguru import logger

from prompt.memory_prompt.chat_bot.prompt_zh.unified_profile_episodic_state import (
    EXTRACT_MEMORY_PROMPT,
)
from src.hook.llm_memory_extract_verify import (
    VALID_MEMORY_CATEGORIES_BY_TYPE,
    is_valid_memory_category,
)
from src.llm.openai_llm import LLMApi
from config import Config


class UnifiedMemoryExtractor:
    """一次 LLM 调用完成 profile / episodic / state 三类记忆的抽取。

    相比 MultiKindLLMExtractor（三次并发调用），本抽取器：
      - 减少 LLM 调用次数（1 次 vs 3 次）
      - 上下文完整共享，避免同一信息被多个 prompt 重复记录
      - 由模型在同一轮输出中完成分类，降低跨类型不一致风险
    """

    MEMORY_TYPES = ("profile", "episodic", "state")

    def __init__(self, llm: Optional[LLMApi] = None):
        self.llm = llm or LLMApi.from_params(
            api_key=Config.llm.API_KEY,
            base_url=Config.llm.BASE_URL,
            model=Config.llm.MODEL,
            temperature=Config.llm.TEMPERATURE,
            max_tokens=Config.llm.MAX_TOKENS,
            top_p=Config.llm.TOP_P,
            timeout=Config.llm.TIMEOUT,
        )

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------
    def extract_all(self, text: str) -> Dict[str, List[Dict]]:
        """对一段主题文本做统一记忆抽取，返回按 memory_type 分组的结果。

        返回格式与 MultiKindLLMExtractor.extract_all 兼容：
            {"profile": [...], "episodic": [...], "state": [...]}
        """
        return self.extract_all_with_context(text)["memories"]

    def extract_all_with_context(self, text: str) -> Dict[str, object]:
        """统一记忆抽取，返回 {"topic_context": str, "memories": {kind: [...]}}。

        `topic_context` 为本段对话的主题摘要：仅当抽取出记忆时才有值，
        否则为空字符串。
        """
        results: Dict[str, List[Dict]] = {k: [] for k in self.MEMORY_TYPES}
        if not text or not text.strip():
            return {"topic_context": "", "memories": results}

        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": EXTRACT_MEMORY_PROMPT.format(text=text)}],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.warning(f"[UnifiedMemoryExtractor] LLM 调用失败: {e}")
            return {"topic_context": "", "memories": results}

        topic_context, memories = self._parse_response(response)
        for mem in memories:
            kind = str(mem.get("memory_type") or "").strip().lower()
            if kind not in results:
                continue
            validated = self._validate_memory(kind, mem)
            if validated is not None:
                results[kind].append(validated)

        # 无记忆时不保留摘要
        has_memory = any(results[k] for k in self.MEMORY_TYPES)
        if not has_memory:
            topic_context = ""

        return {"topic_context": topic_context, "memories": results}

    # ------------------------------------------------------------------
    # 解析
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_response(response: str) -> tuple[str, List[Dict]]:
        """解析 LLM 输出，返回 (topic_context 摘要, 记忆列表)。"""
        if not response:
            return "", []
        text = response.strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                return "", []
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return "", []

        if isinstance(data, dict) and "memories" in data:
            raw = data["memories"]
            memories = [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []
            # prompt 输出字段名为 `topic`，兼容历史 `topic_context`
            ctx = data.get("topic")
            if not isinstance(ctx, str):
                ctx = data.get("topic_context")
            topic_context = ctx.strip() if isinstance(ctx, str) else ""
            return topic_context, memories
        elif isinstance(data, list):
            return "", [item for item in data if isinstance(item, dict)]
        return "", []

    # ------------------------------------------------------------------
    # 校验单条记忆
    # ------------------------------------------------------------------
    @classmethod
    def _validate_memory(cls, kind: str, memory: Dict) -> Optional[Dict]:
        content = str(memory.get("content") or memory.get("memory_content") or "").strip()
        if not content:
            return None

        memory_type = str(memory.get("memory_type") or kind).strip().lower() or kind
        memory_category = str(memory.get("memory_category") or "").strip().lower()

        if memory_type not in cls.MEMORY_TYPES:
            return None

        valid_set = VALID_MEMORY_CATEGORIES_BY_TYPE.get(memory_type, set())
        if valid_set and memory_category not in valid_set:
            return None
        if not is_valid_memory_category(memory_type, memory_category):
            return None

        cite = memory.get("cite")
        if isinstance(cite, list):
            cite = [c for c in cite if isinstance(c, int)]
        else:
            cite = []

        return {
            "memory_content": content,
            "memory_type": memory_type,
            "memory_category": memory_category,
            "importance": cls._safe_float(memory.get("importance"), default=0.5),
            "confidence": cls._safe_float(memory.get("confidence"), default=0.5),
            "cite": cite,
        }

    @staticmethod
    def _safe_float(value, default: float = 0.5) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        return max(0.0, min(1.0, value))


if __name__ == "__main__":
    extractor = UnifiedMemoryExtractor()
    text = """
[0] speaker:user,content:咱们先把这些航班订好吧！接下来，您能否推荐一些坦帕的住宿，位置便利，方便游览景点，并且符合我单人入住的预算？
[1] speaker:ai,content:绝佳选择！对于单人入住且注重便利性的人来说，坦帕市中心是理想之选。我推荐以下房源：\n- 位于纽约市的宽敞一卧公寓（整套公寓，每晚 547 美元，至少入住一晚，评分 5.0）\n- 主要信息：靠近河滨步道（步行即可到达众多景点），禁止吸烟/举办派对，可容纳两人（空间充足）——两晚总价：1,094 美元。\n这样，您的剩余预算为 1,800 美元 - 360 美元（机票）- 1,094 美元（住宿）= 346 美元，可用于餐饮、景点和当地交通。这个住宿方案符合您的需求吗？
[2] speaker:user,content:是的，价格有点贵，我喜欢靠海边的
[3] speaker:ai,content:了解啦！坦帕的海滨风光非常迷人，您可以考虑以下几个住宿选择：\n- 靠近海滨的酒店（如 Hilton Tampa Downtown）：步行即可到达美丽的河滨步道，同时享受城市中心的便利设施。\n- 海滨度假村（如 Marriott Tampa Waterside）：提供宁静的住宿环境，同时靠近海滩和各种娱乐活动。\n- 市中心的高档公寓（如 The St. Petersburg Hotel）：位置便利，靠近购物和餐饮场所，同时提供舒适的住宿环境。\n您对这些选项感兴趣吗？
[4] speaker:user,content:是的，听上去不错，价格呢？
[5] speaker:ai,content:这些住宿的价格范围从每晚 150 美元到 300 美元不等，具体取决于您选择的房型和设施。
[6] speaker:user,content:好的，那我就选择靠近海滨的酒店吧。
    """
    results = extractor.extract_all(text)
    print(results)
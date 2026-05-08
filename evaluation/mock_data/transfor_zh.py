import json
import os
import re
import sys
import time
from typing import Any

from openai import OpenAI


DEFAULT_INPUT = r"D:\aiworks\code\memS\evaluation\benchmark\locomo\data\locomo10.json"
DEFAULT_OUTPUT = r"D:\aiworks\code\memS\evaluation\benchmark\locomo\data\locomo10_zh.json"
DEFAULT_ENV = r"D:\aiworks\code\memS\.env"
DEFAULT_BASE_URL = "https://api.deepseek.com/v1/chat/completions"


def load_env_like(path: str) -> tuple[str, str, str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f".env file not found: {path}")

    key = ""
    model = ""
    base_url = DEFAULT_BASE_URL
    raw_tokens: list[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                k_upper = k.upper()
                v = v.strip().strip("\"' ")
                if k_upper in {"DEEPSEEK_API_KEY", "OPENAI_API_KEY", "API_KEY"} and v:
                    key = v
                if k_upper in {"DEEPSEEK_MODEL", "MODEL"} and v:
                    model = v
                if k_upper in {"DEEPSEEK_BASE_URL", "BASE_URL"} and v:
                    base_url = v
            else:
                raw_tokens.append(line)

    if not key and raw_tokens:
        key = raw_tokens[0]
    if not model and len(raw_tokens) > 1:
        model = raw_tokens[1]
    if not model:
        model = "deepseek-chat"

    if not key:
        raise ValueError("Cannot find DeepSeek API key in .env")
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    # OpenAI SDK expects API root, DeepSeek commonly uses https://api.deepseek.com
    if not normalized.endswith("/v1") and not normalized.endswith(".com"):
        normalized = normalized + "/v1"
    base_url = normalized
    return key, model, base_url


def should_skip_translation(text: str, parent_key: str | None) -> bool:
    if not text or not text.strip():
        return True
    if parent_key in {"sample_id", "evidence"}:
        return True
    if re.fullmatch(r"D\d+:\d+", text.strip()):
        return True
    return False


def collect_strings(node: Any, parent_key: str | None = None, bucket: list[str] | None = None) -> list[str]:
    if bucket is None:
        bucket = []

    if isinstance(node, dict):
        for k, v in node.items():
            collect_strings(v, parent_key=k, bucket=bucket)
    elif isinstance(node, list):
        for item in node:
            collect_strings(item, parent_key=parent_key, bucket=bucket)
    elif isinstance(node, str):
        if not should_skip_translation(node, parent_key):
            bucket.append(node)

    return bucket


def parse_json_array(content: str) -> list[str]:
    content = content.strip()
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except json.JSONDecodeError:
        pass

    m = re.search(r"\[[\s\S]*\]", content)
    if not m:
        raise ValueError("Model output is not a JSON array.")
    parsed = json.loads(m.group(0))
    if not isinstance(parsed, list):
        raise ValueError("Parsed output is not a list.")
    return [str(x) for x in parsed]


def call_deepseek_translate(
    batch: list[str], client: OpenAI, model: str
) -> list[str]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是数据集翻译助手。将输入英文翻译为简体中文，保持原意。"
                    "保留人名、专有名词、代码、编号、日期数字格式。"
                    "只返回 JSON 数组，不要解释。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "请按顺序翻译下面数组中的每个字符串，并返回等长 JSON 数组：\n"
                    + json.dumps(batch, ensure_ascii=False)
                ),
            },
        ],
        temperature=0,
        stream=False,
    )
    content = response.choices[0].message.content or ""
    translated = parse_json_array(content)
    if len(translated) != len(batch):
        raise RuntimeError(
            f"Translated size mismatch: expected {len(batch)}, got {len(translated)}"
        )
    return translated


def chunk_strings(strings: list[str], max_items: int = 24, max_chars: int = 5000) -> list[list[str]]:
    chunks: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for s in strings:
        s_chars = len(s)
        if current and (len(current) >= max_items or current_chars + s_chars > max_chars):
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(s)
        current_chars += s_chars
    if current:
        chunks.append(current)
    return chunks


def translate_chunk_with_fallback(
    chunk: list[str], client: OpenAI, model: str, chunk_label: str
) -> list[str]:
    for attempt in range(1, 6):
        try:
            translated = call_deepseek_translate(chunk, client=client, model=model)
            print(f"{chunk_label} translated {len(chunk)} strings", flush=True)
            return translated
        except Exception as e:  # noqa: BLE001
            sleep_s = min(2 ** attempt, 20)
            print(
                f"{chunk_label} attempt {attempt} failed: {e}; retry in {sleep_s}s",
                flush=True,
            )
            time.sleep(sleep_s)
    print(f"{chunk_label} all retries failed, use original text", flush=True)
    return chunk[:]


def update_mapping_for_node(
    node: Any, mapping: dict[str, str], client: OpenAI, model: str, item_index: int
) -> None:
    collected = collect_strings(node)
    unique_strings = list(dict.fromkeys(collected))
    unknown = [s for s in unique_strings if s not in mapping]
    if not unknown:
        print(f"[item {item_index}] no new strings to translate", flush=True)
        return

    chunks = chunk_strings(unknown)
    total = len(chunks)
    for idx, chunk in enumerate(chunks, start=1):
        label = f"[item {item_index}][chunk {idx}/{total}]"
        translated = translate_chunk_with_fallback(
            chunk=chunk,
            client=client,
            model=model,
            chunk_label=label,
        )
        for src, dst in zip(chunk, translated):
            mapping[src] = dst


def apply_translation(node: Any, mapping: dict[str, str], parent_key: str | None = None) -> Any:
    if isinstance(node, dict):
        return {k: apply_translation(v, mapping, parent_key=k) for k, v in node.items()}
    if isinstance(node, list):
        return [apply_translation(item, mapping, parent_key=parent_key) for item in node]
    if isinstance(node, str):
        if should_skip_translation(node, parent_key):
            return node
        return mapping.get(node, node)
    return node


def main() -> int:
    input_path = DEFAULT_INPUT
    output_path = DEFAULT_OUTPUT
    env_path = DEFAULT_ENV

    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    if len(sys.argv) > 3:
        env_path = sys.argv[3]

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Env: {env_path}")

    api_key, model, base_url = load_env_like(env_path)
    print(f"Model: {model}")
    print(f"Base URL: {base_url}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    client = OpenAI(api_key=api_key, base_url=base_url)
    mapping: dict[str, str] = {}

    if isinstance(data, list):
        total_items = len(data)
        print(f"Total items: {total_items}", flush=True)
        with open(output_path, "w", encoding="utf-8") as out:
            out.write("[\n")
            for i, item in enumerate(data):
                update_mapping_for_node(
                    node=item,
                    mapping=mapping,
                    client=client,
                    model=model,
                    item_index=i + 1,
                )
                translated_item = apply_translation(item, mapping)
                item_text = json.dumps(translated_item, ensure_ascii=False, indent=2)
                item_text = "\n".join("  " + line if line else line for line in item_text.splitlines())
                out.write(item_text)
                if i < total_items - 1:
                    out.write(",\n")
                else:
                    out.write("\n")
                out.flush()
                print(
                    f"[item {i + 1}/{total_items}] written to {output_path}",
                    flush=True,
                )
            out.write("]\n")
            out.flush()
    else:
        # Non-list JSON still follows: translate then write once.
        update_mapping_for_node(node=data, mapping=mapping, client=client, model=model, item_index=1)
        translated_data = apply_translation(data, mapping)
        with open(output_path, "w", encoding="utf-8") as out:
            json.dump(translated_data, out, ensure_ascii=False, indent=2)
            out.flush()
        print(f"[single object] written to {output_path}", flush=True)

    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

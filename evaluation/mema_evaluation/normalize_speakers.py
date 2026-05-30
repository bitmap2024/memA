"""规范化已生成数据集的 speaker 字段，把 speaker_a/speaker_b 占位符替换为人物姓名。

用法：
    python normalize_speakers.py [<file_or_dir> ...]
默认处理 couple_dataset 目录下所有 *.json。
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any


DEFAULT_DIR = r"D:\aiworks\code\memS\evaluation\mock_data\couple_dataset"


def normalize_one(sample: dict[str, Any]) -> tuple[int, int]:
    speaker_a = sample["speaker_a_persona"]["name"]
    speaker_b = sample["speaker_b_persona"]["name"]
    conversation = sample.get("conversation", {})

    fixed = 0
    total = 0
    for key, value in conversation.items():
        if not isinstance(value, list):
            continue
        for turn in value:
            if not isinstance(turn, dict) or "speaker" not in turn:
                continue
            total += 1
            spk = str(turn["speaker"]).strip()
            spk_norm = spk.lower().replace(" ", "").replace("_", "")
            if spk_norm in {"speakera", "a", "male", "boy", "boyfriend"}:
                turn["speaker"] = speaker_a
                fixed += 1
            elif spk_norm in {"speakerb", "b", "female", "girl", "girlfriend", "ai"}:
                turn["speaker"] = speaker_b
                fixed += 1
            elif spk == speaker_a or spk == speaker_b:
                pass
            else:
                print(f"  [warn] 未识别 speaker={spk}，跳过")
    return fixed, total


def process_path(path: str) -> None:
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if name.endswith(".json"):
                process_path(os.path.join(path, name))
        return
    if not path.endswith(".json"):
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        total_fixed = 0
        total_all = 0
        for sample in data:
            f_, t_ = normalize_one(sample)
            total_fixed += f_
            total_all += t_
    else:
        total_fixed, total_all = normalize_one(data)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"{path} -> fixed {total_fixed}/{total_all} speaker entries")


def main() -> int:
    paths = sys.argv[1:] or [DEFAULT_DIR]
    for p in paths:
        process_path(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

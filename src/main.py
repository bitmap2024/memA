#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""memA 命令行入口：extract / update / retrieve / show-config。

示例:
    # 抽取一段对话
    python -m src.main extract --user-id user_001 --history-file sample.json

    # 对用户进行 sleep mode 合并 + 生成 category.md
    python -m src.main update --user-id user_001

    # 在线检索
    python -m src.main retrieve --user-id user_001 --query "用户喜欢什么颜色" --top-k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.setting import Config  # noqa: E402
from src.services.memory_service import MemoryService  # noqa: E402


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def cmd_extract(args: argparse.Namespace) -> int:
    service = MemoryService()
    if args.history_file:
        history = json.loads(Path(args.history_file).read_text(encoding="utf-8"))
    else:
        if not args.conversation_json:
            print("必须传入 --history-file 或 --conversation-json", file=sys.stderr)
            return 1
        history = {
            "user_id": args.user_id,
            "session_id": args.session_id,
            "conversation_date_time": args.conversation_date_time,
            "conversation": json.loads(args.conversation_json),
        }
    history.setdefault("user_id", args.user_id)
    result = service.extract_pipeline.extract_pipeline(
        history=history, compress_rate=args.compress_rate
    )
    _print_json(result)
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    service = MemoryService()
    report = service.update_user_memory(
        user_id=args.user_id,
        max_workers=args.max_workers,
        enable_merge=not args.skip_merge,
        enable_doc=not args.skip_doc,
    )
    _print_json(report)
    return 0


def cmd_retrieve(args: argparse.Namespace) -> int:
    service = MemoryService()
    results = service.retrieve_memory(
        user_id=args.user_id,
        query=args.query,
        top_k=args.top_k,
        memory_type=args.memory_type,
        memory_category=args.memory_category,
        use_bge_rerank=not args.no_bge_rerank,
        use_llm_rerank=args.llm_rerank,
        use_mmr=not args.no_mmr,
    )
    _print_json(results)
    return 0


def cmd_show_config(_: argparse.Namespace) -> int:
    print(Config.describe())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memA")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_extract = sub.add_parser("extract", help="抽取一段对话的记忆")
    p_extract.add_argument("--user-id", required=True)
    p_extract.add_argument("--history-file", help="json 文件路径")
    p_extract.add_argument(
        "--conversation-json", help="直接传入 conversation 数组的 JSON 字符串"
    )
    p_extract.add_argument("--session-id", default=None)
    p_extract.add_argument("--conversation-date-time", default=None)
    p_extract.add_argument("--compress-rate", type=float, default=None)
    p_extract.set_defaults(func=cmd_extract)

    p_update = sub.add_parser("update", help="对用户跑 sleep mode 合并 + 生成 category.md")
    p_update.add_argument("--user-id", required=True)
    p_update.add_argument("--max-workers", type=int, default=4)
    p_update.add_argument("--skip-merge", action="store_true")
    p_update.add_argument("--skip-doc", action="store_true")
    p_update.set_defaults(func=cmd_update)

    p_retrieve = sub.add_parser("retrieve", help="混合检索")
    p_retrieve.add_argument("--user-id", required=True)
    p_retrieve.add_argument("--query", required=True)
    p_retrieve.add_argument("--top-k", type=int, default=10)
    p_retrieve.add_argument("--memory-type", default=None)
    p_retrieve.add_argument("--memory-category", default=None)
    p_retrieve.add_argument("--no-bge-rerank", action="store_true")
    p_retrieve.add_argument("--llm-rerank", action="store_true")
    p_retrieve.add_argument("--no-mmr", action="store_true")
    p_retrieve.set_defaults(func=cmd_retrieve)

    p_cfg = sub.add_parser("show-config", help="打印当前配置")
    p_cfg.set_defaults(func=cmd_show_config)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

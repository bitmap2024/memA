#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CLI entrypoint for the Argo offline memory update CronWorkflow."""

from __future__ import annotations

import argparse
import json

from config.config import Config
from services.update_memory_service import UpdateMemoryService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run offline memory update")
    parser.add_argument("--child-id", default=None)
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--score-threshold", type=float, default=0.9)
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--keep-top-n", type=int, default=10)
    parser.add_argument("--enable-merge", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    service = UpdateMemoryService(config=Config)
    result = service.offline_update_memories(
        child_id=args.child_id,
        agent_id=args.agent_id,
        score_threshold=args.score_threshold,
        max_workers=args.max_workers,
        top_k=args.top_k,
        keep_top_n=args.keep_top_n,
        enable_merge=args.enable_merge,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

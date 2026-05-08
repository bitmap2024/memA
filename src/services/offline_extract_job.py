#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CLI entrypoint for the Argo offline memory extraction CronWorkflow."""

from __future__ import annotations

import argparse
import json

from config.config import Config
from services.extract_memory_service import ExtractMemoryService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run offline memory extraction")
    parser.add_argument("--time-range-hours", "--time_range", type=int, default=Config.memory.OFFLINE_HOUR_RANGE)
    parser.add_argument("--chunk-tokens", type=int, default=2048)
    parser.add_argument("--compress-rate", type=float, default=0.5)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=10)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    service = ExtractMemoryService(config=Config)
    result = service.offline_extract_memories(
        time_range_hours=args.time_range_hours,
        chunk_tokens=args.chunk_tokens,
        compress_rate=args.compress_rate,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not result.get("errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())

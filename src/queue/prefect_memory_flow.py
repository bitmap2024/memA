#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prefect 版记忆抽取 Flow —— 与 task_queue.py 对照学习。

对应 memory_extract_pipeline.extract_and_store 的步骤：
    validate → compress → topic_segment → llm_extract → persist

运行（项目根目录 memA/）::

    pip install prefect>=3.0.0
    python -m src.queue.prefect_memory_flow --dry-run
    python -m src.queue.prefect_memory_flow --dry-run --mode both
    python -m src.queue.prefect_memory_flow --sample deployment/sample_history.json

Dashboard（另开终端）::

    python -m src.queue.prefect_dashboard
    # 浏览器 → http://127.0.0.1:8765

对比要点：
    - TaskQueue：手动 submit / wait_one 串步骤，进程内线程池
    - Prefect：@flow 写编排，@task 写每步，自带重试与运行记录
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

try:
    from prefect import flow, task
except ImportError as exc:
    raise ImportError(
        "请先安装 Prefect: pip install 'prefect>=3.0.0'"
    ) from exc

from src.queue.run_store import RunStore, get_run_store
from src.queue.task_queue import TaskType, build_memory_extract_queue

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE = ROOT / "deployment" / "sample_history.json"


def _lazy_memory_pipeline():
    """延迟导入，避免 dry-run 模式依赖 qdrant / embedding 等重包。"""
    from src.extract_mem.memory_extract_pipeline import MemoryExtractPipeline

    return MemoryExtractPipeline


# ---------------------------------------------------------------------------
# dry-run 用的轻量工具（不 import MemoryExtractPipeline）
# ---------------------------------------------------------------------------
def _now_utc_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _validate_history(history: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(history, dict):
        raise ValueError("history 必须是 dict")

    user_id = str(history.get("user_id") or "").strip()
    if not user_id:
        raise ValueError("history.user_id 必填")

    conversation = history.get("conversation") or []
    if not isinstance(conversation, list) or not conversation:
        raise ValueError("history.conversation 必须是非空 list")

    normalized = []
    for msg in conversation:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").strip().lower()
        content = str(msg.get("content") or "").strip()
        if role not in ("user", "assistant", "system"):
            continue
        if not content:
            continue
        normalized.append({"role": role, "content": content})

    if not normalized:
        raise ValueError("conversation 中没有有效消息")

    return {
        "user_id": user_id,
        "session_id": str(history.get("session_id") or "").strip() or None,
        "conversation_date_time": str(history.get("conversation_date_time") or "").strip()
        or _now_utc_iso(),
        "conversation": normalized,
    }


def _format_topic_text(topic_messages: List[Dict[str, str]]) -> str:
    lines = []
    for idx, msg in enumerate(topic_messages):
        role = msg.get("role", "user").lower()
        content = (msg.get("content") or "").strip().replace("\n", " ")
        if not content:
            continue
        lines.append(f"[{idx}] {role}: {content}")
    return "\n".join(lines)


def _session_id_from_history(history: Dict[str, Any]) -> str:
    import hashlib

    raw = (
        f"{history.get('user_id')}|{history.get('conversation_date_time')}"
        f"|{len(history.get('conversation') or [])}"
    )
    return "sess_" + hashlib.md5(raw.encode("utf-8")).hexdigest()[:20]

# ---------------------------------------------------------------------------
# dry-run：不连 LLM / DB / embedding，只看编排差异
# ---------------------------------------------------------------------------
class _DryRunPipeline:
    """模拟 MemoryExtractPipeline，本地零依赖演示用。"""

    validate_history = staticmethod(_validate_history)
    format_topic_text = staticmethod(_format_topic_text)
    _session_id_from_history = staticmethod(_session_id_from_history)

    def compress_messages(
        self,
        messages: List[Dict[str, str]],
        rate: Optional[float] = None,
    ) -> List[Dict[str, str]]:
        return [
            {**m, "content": f"{m['content']} [compressed]"}
            for m in messages
        ]

    def segment_topics(
        self,
        messages: List[Dict[str, str]],
    ) -> List[List[Dict[str, str]]]:
        mid = max(1, len(messages) // 2)
        if mid >= len(messages):
            return [messages]
        return [messages[:mid], messages[mid:]]

    class _Extractor:
        @staticmethod
        def extract_all(topic_text: str) -> Dict[str, List[Dict[str, Any]]]:
            n = max(1, len(topic_text) // 80)
            return {
                "profile": [
                    {
                        "memory_type": "profile",
                        "memory_category": "preference",
                        "memory_content": f"dry-run profile #{i}",
                        "importance": 0.6,
                        "confidence": 0.8,
                    }
                    for i in range(n)
                ],
                "episodic": [],
                "state": [
                    {
                        "memory_type": "state",
                        "memory_category": "emotion",
                        "memory_content": "dry-run state",
                        "importance": 0.5,
                        "confidence": 0.7,
                    }
                ],
            }

    extractor = _Extractor()

    def _persist_memories(
        self,
        user_id: str,
        session_id: str,
        session_time: str,
        memories: List[Dict[str, Any]],
        topic_idx: int,
    ) -> Tuple[int, int]:
        return len(memories), 0


_pipeline_cache: Dict[bool, Any] = {}


def _get_pipeline(dry_run: bool) -> Any:
    if dry_run not in _pipeline_cache:
        _pipeline_cache[dry_run] = (
            _DryRunPipeline() if dry_run else _lazy_memory_pipeline()()
        )
    return _pipeline_cache[dry_run]


def _track_step(store: RunStore, run_id: str, name: str):
    """记录单步开始/结束，供 dashboard 展示。"""

    class _Tracker:
        def __enter__(self):
            store.step(run_id, name, "running")
            return self

        def __exit__(self, exc_type, exc, _tb):
            if exc_type is None:
                store.step(run_id, name, "completed")
            else:
                store.step(run_id, name, "failed", detail=str(exc))
            return False

    return _Tracker()


# ---------------------------------------------------------------------------
# Prefect Task = Activity（每一步具体工作）
# ---------------------------------------------------------------------------
@task(name="1-validate", retries=1, retry_delay_seconds=2)
def task_validate(history: Dict[str, Any]) -> Dict[str, Any]:
    return _validate_history(history)


@task(name="2-compress", retries=2, retry_delay_seconds=3)
def task_compress(
    messages: List[Dict[str, str]],
    dry_run: bool,
    compress_rate: Optional[float] = None,
) -> List[Dict[str, str]]:
    return _get_pipeline(dry_run).compress_messages(messages, rate=compress_rate)


@task(name="3-topic-segment", retries=2, retry_delay_seconds=3)
def task_topic_segment(
    messages: List[Dict[str, str]],
    dry_run: bool,
) -> List[List[Dict[str, str]]]:
    return _get_pipeline(dry_run).segment_topics(messages)


@task(name="4-llm-extract", retries=3, retry_delay_seconds=5)
def task_llm_extract(topic_text: str, dry_run: bool) -> Dict[str, List[Dict[str, Any]]]:
    return _get_pipeline(dry_run).extractor.extract_all(topic_text)


@task(name="5-persist", retries=2, retry_delay_seconds=3)
def task_persist(
    user_id: str,
    session_id: str,
    session_time: str,
    memories: List[Dict[str, Any]],
    topic_idx: int,
    dry_run: bool,
) -> Tuple[int, int]:
    return _get_pipeline(dry_run)._persist_memories(
        user_id=user_id,
        session_id=session_id,
        session_time=session_time,
        memories=memories,
        topic_idx=topic_idx,
    )


# ---------------------------------------------------------------------------
# Prefect Flow = Workflow（编排剧本，不写重计算）
# ---------------------------------------------------------------------------
@flow(name="extract-memory", log_prints=True)
def extract_memory_flow(
    history: Dict[str, Any],
    *,
    dry_run: bool = False,
    compress_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """与 MemoryExtractPipeline.extract_and_store 等价的 Prefect 编排。"""
    store = get_run_store()
    run = store.begin(
        runner="prefect",
        user_id=str(history.get("user_id") or ""),
        session_id=str(history.get("session_id") or ""),
        dry_run=dry_run,
    )

    try:
        pipeline = _get_pipeline(dry_run)

        with _track_step(store, run.id, "1-validate"):
            cleaned = task_validate(history)
        user_id = cleaned["user_id"]
        session_id = cleaned["session_id"] or pipeline._session_id_from_history(cleaned)
        session_time = cleaned["conversation_date_time"]

        with _track_step(store, run.id, "2-compress"):
            compressed = task_compress(cleaned["conversation"], dry_run, compress_rate)
        with _track_step(store, run.id, "3-topic-segment"):
            topics = task_topic_segment(compressed, dry_run)

        report: Dict[str, Any] = {
            "runner": "prefect",
            "dry_run": dry_run,
            "user_id": user_id,
            "session_id": session_id,
            "topics": len(topics),
            "extracted": {"profile": 0, "episodic": 0, "state": 0},
            "stored": 0,
            "skipped_duplicates": 0,
        }

        for topic_idx, topic in enumerate(topics):
            topic_text = pipeline.format_topic_text(topic)
            if not topic_text:
                continue

            with _track_step(store, run.id, "4-llm-extract"):
                kind_to_memories = task_llm_extract(topic_text, dry_run)
            for kind, mems in kind_to_memories.items():
                report["extracted"][kind] += len(mems)

            with _track_step(store, run.id, "5-persist"):
                for kind, mems in kind_to_memories.items():
                    if not mems:
                        continue
                    stored, dup = task_persist(
                        user_id, session_id, session_time, mems, topic_idx, dry_run
                    )
                    report["stored"] += stored
                    report["skipped_duplicates"] += dup

        store.finish(run.id, "completed", report)
        logger.info(f"[PrefectFlow] 抽取完成: {report}")
        return report
    except Exception as e:
        store.finish(run.id, "failed", error=str(e))
        raise


# ---------------------------------------------------------------------------
# 对照：同一 pipeline 用 TaskQueue（线程池）跑一遍
# ---------------------------------------------------------------------------
def extract_memory_with_task_queue(
    history: Dict[str, Any],
    *,
    dry_run: bool = False,
    compress_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """与 extract_memory_flow 等价逻辑，但用手动 submit/wait 串步骤。"""
    store = get_run_store()
    run = store.begin(
        runner="task_queue",
        user_id=str(history.get("user_id") or ""),
        session_id=str(history.get("session_id") or ""),
        dry_run=dry_run,
    )

    try:
        pipeline = _get_pipeline(dry_run)
        queue = build_memory_extract_queue(pipeline, max_workers=4)

        with _track_step(store, run.id, "1-validate"):
            cleaned = _validate_history(history)
        user_id = cleaned["user_id"]
        session_id = cleaned["session_id"] or pipeline._session_id_from_history(cleaned)
        session_time = cleaned["conversation_date_time"]

        with _track_step(store, run.id, "2-compress"):
            compress_id = queue.submit(
                TaskType.COMPRESS,
                {"messages": cleaned["conversation"], "rate": compress_rate},
                user_id=user_id,
                session_id=session_id,
            )
            compressed = queue.wait_one(compress_id).result

        with _track_step(store, run.id, "3-topic-segment"):
            segment_id = queue.submit(
                TaskType.TOPIC_SEGMENT,
                {"messages": compressed},
                user_id=user_id,
                session_id=session_id,
            )
            topics = queue.wait_one(segment_id).result

        report: Dict[str, Any] = {
            "runner": "task_queue",
            "dry_run": dry_run,
            "user_id": user_id,
            "session_id": session_id,
            "topics": len(topics),
            "extracted": {"profile": 0, "episodic": 0, "state": 0},
            "stored": 0,
            "skipped_duplicates": 0,
        }

        extract_ids: List[str] = []
        topic_texts: List[str] = []
        for topic in topics:
            topic_text = pipeline.format_topic_text(topic)
            if not topic_text:
                continue
            topic_texts.append(topic_text)
            extract_ids.append(
                queue.submit(
                    TaskType.LLM_EXTRACT,
                    {"topic_text": topic_text},
                    user_id=user_id,
                    session_id=session_id,
                )
            )

        with _track_step(store, run.id, "4-llm-extract"):
            extract_tasks = queue.wait_all(extract_ids)
        for topic_idx, (_topic_text, ext_task) in enumerate(zip(topic_texts, extract_tasks)):
            if ext_task.status.value != "done":
                logger.warning(f"LLM 抽取失败: {ext_task.error}")
                continue
            kind_to_memories = ext_task.result
            for kind, mems in kind_to_memories.items():
                report["extracted"][kind] += len(mems)

            with _track_step(store, run.id, "5-persist"):
                for kind, mems in kind_to_memories.items():
                    if not mems:
                        continue
                    stored, dup = pipeline._persist_memories(
                        user_id=user_id,
                        session_id=session_id,
                        session_time=session_time,
                        memories=mems,
                        topic_idx=topic_idx,
                    )
                    report["stored"] += stored
                    report["skipped_duplicates"] += dup

        queue.shutdown()
        store.finish(run.id, "completed", report)
        logger.info(f"[TaskQueue] 抽取完成: {report}")
        return report
    except Exception as e:
        store.finish(run.id, "failed", error=str(e))
        raise


def _load_history(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _print_compare(prefect_report: Dict[str, Any], queue_report: Dict[str, Any]) -> None:
    print("\n========== Prefect vs TaskQueue 对比 ==========")
    print(f"{'字段':<22} {'Prefect':>12} {'TaskQueue':>12}")
    print("-" * 48)
    for key in ("topics", "stored", "skipped_duplicates"):
        print(f"{key:<22} {prefect_report[key]:>12} {queue_report[key]:>12}")
    for kind in ("profile", "episodic", "state"):
        pk = prefect_report["extracted"][kind]
        qk = queue_report["extracted"][kind]
        print(f"extracted.{kind:<12} {pk:>12} {qk:>12}")
    print("\n代码风格差异：")
    print("  Prefect    → @flow 里直接写 for/await task_xxx()，框架记日志与重试")
    print("  TaskQueue  → 每步 queue.submit + wait_one，编排和调度混在一起")
    print("==============================================\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prefect 记忆抽取最小示例")
    parser.add_argument(
        "--sample",
        type=Path,
        default=DEFAULT_SAMPLE,
        help="history JSON 路径",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="不连 LLM/DB/embedding，用 stub 演示编排",
    )
    parser.add_argument(
        "--mode",
        choices=("prefect", "threadpool", "both"),
        default="prefect",
        help="prefect=只跑 Flow; threadpool=只跑 TaskQueue; both=对比",
    )
    args = parser.parse_args(argv)

    if not args.sample.exists():
        print(f"样本文件不存在: {args.sample}", file=sys.stderr)
        return 1

    history = _load_history(args.sample)
    print(f"加载样本: {args.sample}  dry_run={args.dry_run}  mode={args.mode}")

    if args.mode in ("prefect", "both"):
        prefect_report = extract_memory_flow(history, dry_run=args.dry_run)
        print("Prefect report:", json.dumps(prefect_report, ensure_ascii=False, indent=2))

    if args.mode in ("threadpool", "both"):
        queue_report = extract_memory_with_task_queue(history, dry_run=args.dry_run)
        print("TaskQueue report:", json.dumps(queue_report, ensure_ascii=False, indent=2))

    if args.mode == "both":
        _print_compare(prefect_report, queue_report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

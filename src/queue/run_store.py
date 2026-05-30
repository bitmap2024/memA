#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""本地运行记录存储，供 dashboard 展示（不依赖 Prefect Server）。"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORE_PATH = ROOT / "data" / "queue_runs.json"
MAX_RUNS = 200


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class StepRecord:
    name: str
    status: str  # pending | running | completed | failed
    started_at: str = ""
    finished_at: str = ""
    detail: str = ""


@dataclass
class RunRecord:
    id: str
    runner: str  # prefect | task_queue
    user_id: str = ""
    session_id: str = ""
    dry_run: bool = False
    status: str = "running"  # running | completed | failed
    started_at: str = field(default_factory=_now_iso)
    finished_at: str = ""
    steps: List[StepRecord] = field(default_factory=list)
    report: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


class RunStore:
    """线程安全的 JSON 文件存储。"""

    def __init__(self, path: Path = DEFAULT_STORE_PATH) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _load(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            with self.path.open(encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _save(self, runs: List[Dict[str, Any]]) -> None:
        trimmed = runs[:MAX_RUNS]
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(trimmed, f, ensure_ascii=False, indent=2)

    def begin(
        self,
        *,
        runner: str,
        user_id: str = "",
        session_id: str = "",
        dry_run: bool = False,
    ) -> RunRecord:
        run = RunRecord(
            id=uuid.uuid4().hex[:12],
            runner=runner,
            user_id=user_id,
            session_id=session_id,
            dry_run=dry_run,
        )
        with self._lock:
            runs = self._load()
            runs.insert(0, asdict(run))
            self._save(runs)
        return run

    def step(
        self,
        run_id: str,
        name: str,
        status: str,
        *,
        detail: str = "",
    ) -> None:
        with self._lock:
            runs = self._load()
            for run in runs:
                if run.get("id") != run_id:
                    continue
                steps: List[Dict[str, Any]] = run.setdefault("steps", [])
                existing = next((s for s in steps if s["name"] == name), None)
                now = _now_iso()
                if existing is None:
                    steps.append(
                        {
                            "name": name,
                            "status": status,
                            "started_at": now if status == "running" else "",
                            "finished_at": now if status in ("completed", "failed") else "",
                            "detail": detail,
                        }
                    )
                else:
                    if status == "running" and not existing.get("started_at"):
                        existing["started_at"] = now
                    existing["status"] = status
                    if status in ("completed", "failed"):
                        existing["finished_at"] = now
                    if detail:
                        existing["detail"] = detail
                self._save(runs)
                return

    def finish(
        self,
        run_id: str,
        status: str,
        report: Optional[Dict[str, Any]] = None,
        *,
        error: str = "",
    ) -> None:
        with self._lock:
            runs = self._load()
            for run in runs:
                if run.get("id") != run_id:
                    continue
                run["status"] = status
                run["finished_at"] = _now_iso()
                if report is not None:
                    run["report"] = report
                if error:
                    run["error"] = error
                self._save(runs)
                return

    def list_runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return self._load()[:limit]

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            for run in self._load():
                if run.get("id") == run_id:
                    return run
        return None


_store: Optional[RunStore] = None


def get_run_store() -> RunStore:
    global _store
    if _store is None:
        _store = RunStore()
    return _store

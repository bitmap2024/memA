#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SqliteMemoryStore：记忆层（memories）在本地 SQLite 上的实现。

- 使用 stdlib 的 sqlite3，无需额外依赖；
- 占位符使用 `?`，upsert 使用 `INSERT ... ON CONFLICT(memory_id) DO UPDATE`；
- DDL 直接复用同目录下的 `sql` 文件（SQLite 语法）。

注意：三层（conversations / topics / memories）的事务性入库由
MysqlMemoryStore 负责，SQLite 仅作为本地开发时记忆层的轻量后端。
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from loguru import logger

from config.releatiion_schema import MemoryItem
from config.setting import Config, MysqlConfig
from src.db.base import (
    INSERT_COLUMNS,
    INSERT_COL_COUNT,
    MemoryStoreClient,
    memory_item_to_row,
    now_iso,
    row_to_memory_item,
)


_PLACEHOLDERS = ",".join(["?"] * INSERT_COL_COUNT)
_UPSERT_SQL = (
    f"INSERT INTO memories({INSERT_COLUMNS}) "
    f"VALUES ({_PLACEHOLDERS}) "
    "ON CONFLICT(memory_id) DO UPDATE SET "
    "memory_content=excluded.memory_content, "
    "memory_type=excluded.memory_type, "
    "memory_category=excluded.memory_category, "
    "status=excluded.status, "
    "importance=excluded.importance, "
    "confidence=excluded.confidence, "
    "retrieval_count=excluded.retrieval_count, "
    "last_retrieved_at=excluded.last_retrieved_at, "
    "source_topic_ids=excluded.source_topic_ids, "
    "source_topic_cites=excluded.source_topic_cites, "
    "derived_from_memory_ids=excluded.derived_from_memory_ids, "
    "derived_memory_count=excluded.derived_memory_count, "
    "metadata=excluded.metadata"
)


class SqliteMemoryStore(MemoryStoreClient):
    """纯 SQLite 后端（记忆层，与 MysqlMemoryStore 记忆层语义等价）。"""

    def __init__(self, cfg: Optional[MysqlConfig] = None, sqlite_path: Optional[str] = None):
        self.cfg = cfg or Config.mysql
        self.sqlite_path = Path(sqlite_path or self.cfg.SQLITE_PATH)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        logger.info(f"[SqliteMemoryStore] 启用 SQLite 后端: {self.sqlite_path}")
        self.ensure_schema()

    @property
    def backend(self) -> str:
        return "sqlite"

    # ------------------------------------------------------------------
    # 连接
    # ------------------------------------------------------------------
    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.sqlite_path), timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        try:
            yield conn
        finally:
            conn.close()

    def ensure_schema(self) -> None:
        ddl_path = Path(__file__).resolve().parent / "sql"
        if not ddl_path.exists():
            logger.warning(f"[SqliteMemoryStore] DDL 文件不存在: {ddl_path}")
            return
        sql_text = ddl_path.read_text(encoding="utf-8")
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            for raw_stmt in sql_text.split(";"):
                stmt = raw_stmt.strip()
                if not stmt:
                    continue
                try:
                    cur.execute(stmt)
                except Exception as e:
                    logger.debug(f"[SqliteMemoryStore] DDL 语句跳过(可能已存在): {e}")

    @staticmethod
    def _row_to_dict(row: Any) -> Dict[str, Any]:
        if row is None:
            return {}
        if isinstance(row, sqlite3.Row):
            return {k: row[k] for k in row.keys()}
        return dict(row)

    # ------------------------------------------------------------------
    # MemoryItem CRUD
    # ------------------------------------------------------------------
    def upsert_memory(self, item: MemoryItem) -> None:
        self.upsert_memories([item])

    def upsert_memories(self, items: Sequence[MemoryItem]) -> int:
        if not items:
            return 0
        rows = [memory_item_to_row(it) for it in items]
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.executemany(_UPSERT_SQL, rows)
            conn.commit()
        return len(rows)

    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM memories WHERE memory_id=?", (str(memory_id),))
            row = cur.fetchone()
        if not row:
            return None
        return row_to_memory_item(self._row_to_dict(row))

    def get_memories_by_ids(self, memory_ids: Sequence[str]) -> List[MemoryItem]:
        if not memory_ids:
            return []
        placeholders = ",".join(["?"] * len(memory_ids))
        sql = f"SELECT * FROM memories WHERE memory_id IN ({placeholders})"
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute(sql, tuple(str(i) for i in memory_ids))
            rows = cur.fetchall()
        return [row_to_memory_item(self._row_to_dict(r)) for r in rows]

    def list_user_memories(
        self,
        user_id: str,
        status: Optional[str] = "active",
        memory_type: Optional[str] = None,
        memory_category: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[MemoryItem]:
        sql = "SELECT * FROM memories WHERE user_id=?"
        params: List[Any] = [user_id]
        if status:
            sql += " AND status=?"
            params.append(status)
        if memory_type:
            sql += " AND memory_type=?"
            params.append(memory_type)
        if memory_category:
            sql += " AND memory_category=?"
            params.append(memory_category)
        sql += " ORDER BY created_at DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
        return [row_to_memory_item(self._row_to_dict(row)) for row in rows]

    def update_status(self, memory_ids: Sequence[str], status: str) -> int:
        if not memory_ids:
            return 0
        placeholders = ",".join(["?"] * len(memory_ids))
        sql = f"UPDATE memories SET status=? WHERE memory_id IN ({placeholders})"
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute(sql, (status, *[str(i) for i in memory_ids]))
            conn.commit()
            return cur.rowcount

    def bump_retrieval(self, memory_ids: Sequence[str]) -> int:
        if not memory_ids:
            return 0
        placeholders = ",".join(["?"] * len(memory_ids))
        sql = (
            "UPDATE memories SET retrieval_count = retrieval_count + 1, "
            f"last_retrieved_at = ? WHERE memory_id IN ({placeholders})"
        )
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute(sql, (now_iso(), *[str(i) for i in memory_ids]))
            conn.commit()
            return cur.rowcount

    def delete_memory(self, memory_id: str) -> int:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM memories WHERE memory_id=?", (str(memory_id),))
            conn.commit()
            return cur.rowcount

    def list_user_ids(self) -> List[str]:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT user_id FROM memories")
            rows = cur.fetchall()
        return [self._row_to_dict(r)["user_id"] for r in rows]

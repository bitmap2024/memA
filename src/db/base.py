#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MemoryStore 抽象基类与共享序列化逻辑。

MySQL 与 SQLite 客户端共享同一份业务语义（表名 / 字段 / 行为），
仅在「连接方式 / 占位符 / upsert 语法」上有差异。

外层（services / pipeline / sleep mode 等）通过 `MemoryStoreClient` 接口编程，
不感知具体后端，由 `src/db/__init__.py` 的工厂函数按配置路由。
"""

from __future__ import annotations

import abc
import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from config.releatiion_schema import MemoryItem


class MemoryStoreException(Exception):
    """关系存储层通用异常。"""


def now_iso() -> str:
    """返回当前 UTC 时间，格式兼容 MySQL DATETIME(3)：YYYY-MM-DD HH:MM:SS.mmm"""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d %H:%M:%S.") + f"{now.microsecond // 1000:03d}"


def _parse_json(val: Any, default: Any) -> Any:
    """安全解析 JSON 列：已经是 dict/list 直接返回，str 做 loads，其余返回 default。"""
    if val is None:
        return default
    if isinstance(val, (dict, list)):
        return val
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return default


def row_to_memory_item(row: Dict[str, Any]) -> MemoryItem:
    """将数据库行（dict）反序列化为 MemoryItem（三层 schema 的记忆层）。"""
    return MemoryItem(
        memory_id=str(row.get("memory_id") or ""),
        user_id=str(row.get("user_id") or ""),
        memory_content=str(row.get("memory_content") or ""),
        memory_type=str(row.get("memory_type") or ""),
        memory_category=str(row.get("memory_category") or ""),
        status=str(row.get("status") or "active"),
        created_at=str(row.get("created_at") or ""),
        importance=float(row.get("importance") or 0.0),
        confidence=float(row.get("confidence") or 0.0),
        retrieval_count=int(row.get("retrieval_count") or 0),
        last_retrieved_at=str(row.get("last_retrieved_at") or ""),
        source_topic_ids=_parse_json(row.get("source_topic_ids"), []),
        source_topic_cites=_parse_json(row.get("source_topic_cites"), {}),
        derived_from_memory_ids=_parse_json(row.get("derived_from_memory_ids"), []),
        derived_memory_count=int(row.get("derived_memory_count") or 0),
        metadata=_parse_json(row.get("metadata"), {}),
    )


# INSERT 列顺序，与 memories DDL 定义一致（16 列）
INSERT_COLUMNS = (
    "memory_id, user_id, memory_content, memory_type, memory_category, "
    "status, created_at, importance, confidence, "
    "retrieval_count, last_retrieved_at, "
    "source_topic_ids, source_topic_cites, derived_from_memory_ids, "
    "derived_memory_count, metadata"
)
INSERT_COL_COUNT = 16


def memory_item_to_row(item: MemoryItem) -> Tuple:
    """统一打平为 16 列元组，列顺序与 INSERT_COLUMNS 严格一致。"""
    now = now_iso()
    return (
        item.memory_id,
        item.user_id,
        item.memory_content,
        item.memory_type,
        item.memory_category,
        item.status or "active",
        item.created_at or now,
        float(item.importance or 0.0),
        float(item.confidence or 0.0),
        int(item.retrieval_count or 0),
        item.last_retrieved_at or None,
        json.dumps([str(i) for i in (item.source_topic_ids or [])], ensure_ascii=False),
        json.dumps(
            {str(k): list(v) for k, v in (item.source_topic_cites or {}).items()},
            ensure_ascii=False,
        ),
        json.dumps([str(i) for i in (item.derived_from_memory_ids or [])], ensure_ascii=False),
        int(item.derived_memory_count or 0),
        json.dumps(item.metadata or {}, ensure_ascii=False),
    )


def memory_item_to_dict(item: MemoryItem) -> Dict[str, Any]:
    return asdict(item)


class MemoryStoreClient(abc.ABC):
    """MemoryItem 关系存储客户端接口。

    上层组件只与该接口交互；具体由 MysqlMemoryStore / SqliteMemoryStore 实现。
    """

    # ------------------------------------------------------------------
    # 表 & schema
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def ensure_schema(self) -> None: ...

    # ------------------------------------------------------------------
    # MemoryItem CRUD
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def upsert_memory(self, item: MemoryItem) -> None: ...

    @abc.abstractmethod
    def upsert_memories(self, items: Sequence[MemoryItem]) -> int: ...

    @abc.abstractmethod
    def get_memory(self, memory_id: str) -> Optional[MemoryItem]: ...

    @abc.abstractmethod
    def list_user_memories(
        self,
        user_id: str,
        status: Optional[str] = "active",
        memory_type: Optional[str] = None,
        memory_category: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[MemoryItem]: ...

    @abc.abstractmethod
    def update_status(self, memory_ids: Sequence[str], status: str) -> int: ...

    @abc.abstractmethod
    def bump_retrieval(self, memory_ids: Sequence[str]) -> int: ...

    @abc.abstractmethod
    def delete_memory(self, memory_id: str) -> int: ...
    # ------------------------------------------------------------------
    # 元信息
    # ------------------------------------------------------------------
    @property
    @abc.abstractmethod
    def backend(self) -> str:
        """返回 "mysql" 或 "sqlite"，便于上层日志/排错。"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MysqlMemoryStore：记忆三层关系存储（基于 PyMySQL）。

维护记忆系统的三层关系数据：
  - 第一层 conversations : 原始对话 + 压缩对话
  - 第二层 topics        : 主题（含 memory 时才落库）
  - 第三层 memories       : 记忆

支持两个核心阶段：
  1. 抽取阶段 —— 抽取结束后将 ConversationItem / TopicItem / MemoryItem
     三层数据在同一事务中同时写入（:meth:`insert_hierarchy`）。
  2. 合并阶段 —— sleep mode 下在 memory layer 将多条相似 memory 合并为一条，
     旧 memory 归档（:meth:`merge_memories`）。
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger

try:
    import pymysql  # type: ignore
    from pymysql.cursors import DictCursor  # type: ignore

    _HAS_PYMYSQL = True
except Exception:  # pragma: no cover
    pymysql = None
    DictCursor = None
    _HAS_PYMYSQL = False

from config.releatiion_schema import ConversationItem, MemoryItem, TopicItem
from config.setting import Config, MysqlConfig
from src.utils.snow_id import generate_id


class MemoryStoreException(Exception):
    """关系存储层通用异常。"""


# -----------------------------------------------------------------------------
# 序列化工具
# -----------------------------------------------------------------------------
def _now_iso() -> str:
    """当前 UTC 时间，兼容 MySQL DATETIME(3)：YYYY-MM-DD HH:MM:SS.mmm"""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d %H:%M:%S.") + f"{now.microsecond // 1000:03d}"


def _to_dt(value: Optional[str]) -> Optional[str]:
    """将 ISO 时间字符串规整为 MySQL DATETIME 字符串；空值返回 None。"""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt.microsecond // 1000:03d}"
    except (ValueError, AttributeError):
        return str(value)


def _json(obj: Any) -> str:
    return json.dumps(obj if obj is not None else None, ensure_ascii=False)


def _parse_json(val: Any, default: Any) -> Any:
    if val is None:
        return default
    if isinstance(val, (dict, list)):
        return val
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return default


# ---- conversations ----------------------------------------------------------
CONVERSATION_COLUMNS = (
    "conversation_id, raw_conversation_id, user_id, "
    "conversation_data, conversation_compressed_data, "
    "conversation_date_time, created_at"
)
_CONVERSATION_PLACEHOLDERS = ",".join(["%s"] * 7)
CONVERSATION_UPSERT_SQL = (
    f"INSERT INTO conversations ({CONVERSATION_COLUMNS}) "
    f"VALUES ({_CONVERSATION_PLACEHOLDERS}) "
    "ON DUPLICATE KEY UPDATE "
    "raw_conversation_id=VALUES(raw_conversation_id), "
    "user_id=VALUES(user_id), "
    "conversation_data=VALUES(conversation_data), "
    "conversation_compressed_data=VALUES(conversation_compressed_data), "
    "conversation_date_time=VALUES(conversation_date_time)"
)


def conversation_to_row(item: ConversationItem) -> Tuple:
    now = _now_iso()
    return (
        str(item.conversation_id),
        str(item.raw_conversation_id),
        str(item.user_id or ""),
        _json(item.conversation_data or []),
        _json(item.conversation_compressed_data or []),
        _to_dt(item.conversation_date_time) or now,
        _to_dt(item.created_at) or now,
    )


# ---- topics -----------------------------------------------------------------
TOPIC_COLUMNS = (
    "topic_id, topic_idx, topic_context, topic_messages, status, "
    "source_conversation_ids, derived_from_topic_ids, "
    "derived_topic_count, created_at"
)
_TOPIC_PLACEHOLDERS = ",".join(["%s"] * 9)
TOPIC_UPSERT_SQL = (
    f"INSERT INTO topics ({TOPIC_COLUMNS}) "
    f"VALUES ({_TOPIC_PLACEHOLDERS}) "
    "ON DUPLICATE KEY UPDATE "
    "topic_idx=VALUES(topic_idx), "
    "topic_context=VALUES(topic_context), "
    "topic_messages=VALUES(topic_messages), "
    "status=VALUES(status), "
    "source_conversation_ids=VALUES(source_conversation_ids), "
    "derived_from_topic_ids=VALUES(derived_from_topic_ids), "
    "derived_topic_count=VALUES(derived_topic_count)"
)


def topic_to_row(item: TopicItem) -> Tuple:
    now = _now_iso()
    return (
        str(item.topic_id),
        int(item.topic_idx or 0),
        item.topic_context or "",
        _json(item.topic_messages or []),
        item.status or "active",
        _json([str(i) for i in (item.source_conversation_ids or [])]),
        _json([str(i) for i in (item.derived_from_topic_ids or [])]),
        int(item.derived_topic_count or 0),
        _to_dt(item.created_at) or now,
    )


# ---- memories ---------------------------------------------------------------
MEMORY_COLUMNS = (
    "memory_id, user_id, memory_content, memory_type, memory_category, "
    "status, created_at, importance, confidence, "
    "retrieval_count, last_retrieved_at, "
    "source_topic_ids, source_topic_cites, "
    "derived_from_memory_ids, derived_memory_count, metadata"
)
_MEMORY_PLACEHOLDERS = ",".join(["%s"] * 16)
MEMORY_UPSERT_SQL = (
    f"INSERT INTO memories ({MEMORY_COLUMNS}) "
    f"VALUES ({_MEMORY_PLACEHOLDERS}) "
    "ON DUPLICATE KEY UPDATE "
    "memory_content=VALUES(memory_content), "
    "memory_type=VALUES(memory_type), "
    "memory_category=VALUES(memory_category), "
    "status=VALUES(status), "
    "importance=VALUES(importance), "
    "confidence=VALUES(confidence), "
    "retrieval_count=VALUES(retrieval_count), "
    "last_retrieved_at=VALUES(last_retrieved_at), "
    "source_topic_ids=VALUES(source_topic_ids), "
    "source_topic_cites=VALUES(source_topic_cites), "
    "derived_from_memory_ids=VALUES(derived_from_memory_ids), "
    "derived_memory_count=VALUES(derived_memory_count), "
    "metadata=VALUES(metadata)"
)


def memory_to_row(item: MemoryItem) -> Tuple:
    now = _now_iso()
    return (
        str(item.memory_id),
        str(item.user_id),
        item.memory_content or "",
        item.memory_type or "",
        item.memory_category or "",
        item.status or "active",
        _to_dt(item.created_at) or now,
        float(item.importance or 0.0),
        float(item.confidence or 0.0),
        int(item.retrieval_count or 0),
        _to_dt(item.last_retrieved_at),
        _json([str(i) for i in (item.source_topic_ids or [])]),
        _json({str(k): list(v) for k, v in (item.source_topic_cites or {}).items()}),
        _json([str(i) for i in (item.derived_from_memory_ids or [])]),
        int(item.derived_memory_count or 0),
        _json(item.metadata or {}),
    )


def row_to_memory_item(row: Dict[str, Any]) -> MemoryItem:
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


def _decode_topic_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """把 topics 行的 JSON 列解析为 Python 结构，便于溯源。"""
    row["topic_messages"] = _parse_json(row.get("topic_messages"), [])
    row["source_conversation_ids"] = _parse_json(row.get("source_conversation_ids"), [])
    row["derived_from_topic_ids"] = _parse_json(row.get("derived_from_topic_ids"), [])
    return row


def _decode_conversation_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """把 conversations 行的 JSON 列解析为 Python 结构。"""
    row["conversation_data"] = _parse_json(row.get("conversation_data"), [])
    row["conversation_compressed_data"] = _parse_json(
        row.get("conversation_compressed_data"), []
    )
    return row


# -----------------------------------------------------------------------------
# Store
# -----------------------------------------------------------------------------
class MysqlMemoryStore:
    """记忆三层关系存储后端（MySQL）。"""

    def __init__(self, cfg: Optional[MysqlConfig] = None):
        if not _HAS_PYMYSQL:
            raise MemoryStoreException("MySQL 模式需要 pymysql，请 `pip install pymysql`")
        self.cfg = cfg or Config.mysql
        self._lock = threading.RLock()
        logger.info(
            f"[MysqlMemoryStore] 启用 MySQL 后端: "
            f"{self.cfg.HOST}:{self.cfg.PORT}/{self.cfg.DATABASE}"
        )
        self.ensure_schema()

    @property
    def backend(self) -> str:
        return "mysql"

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------
    @contextmanager
    def _conn(self, autocommit: bool = True):
        conn = pymysql.connect(
            host=self.cfg.HOST,
            port=self.cfg.PORT,
            user=self.cfg.USER,
            password=self.cfg.PASSWORD,
            database=self.cfg.DATABASE,
            charset="utf8mb4",
            cursorclass=DictCursor,
            autocommit=autocommit,
        )
        try:
            yield conn
        finally:
            conn.close()

    def ensure_schema(self) -> None:
        ddl_path = Path(__file__).resolve().parent / "sql"
        if not ddl_path.exists():
            logger.warning(f"[MysqlMemoryStore] DDL 文件不存在: {ddl_path}")
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
                    logger.debug(f"[MysqlMemoryStore] DDL 语句跳过(可能已存在): {e}")

    # ==================================================================
    # 阶段一：抽取入库 —— 三层同时写入
    # ==================================================================
    def insert_hierarchy(
        self,
        conversation_item: ConversationItem,
        topic_items: Optional[Sequence[TopicItem]] = None,
        memory_items: Optional[Sequence[MemoryItem]] = None,
    ) -> Dict[str, int]:
        """在同一事务内写入 conversation / topics / memories 三层数据。

        - conversation 层始终写入；
        - topics / memories 仅在有内容时写入。
        任一层失败则整体回滚，保证三层一致性。
        """
        if conversation_item is None:
            raise MemoryStoreException("insert_hierarchy 需要 conversation_item")

        topic_items = list(topic_items or [])
        memory_items = list(memory_items or [])

        conv_row = conversation_to_row(conversation_item)
        topic_rows = [topic_to_row(t) for t in topic_items]
        memory_rows = [memory_to_row(m) for m in memory_items]

        with self._lock, self._conn(autocommit=False) as conn:
            try:
                cur = conn.cursor()
                cur.execute(CONVERSATION_UPSERT_SQL, conv_row)
                if topic_rows:
                    cur.executemany(TOPIC_UPSERT_SQL, topic_rows)
                if memory_rows:
                    cur.executemany(MEMORY_UPSERT_SQL, memory_rows)
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"[MysqlMemoryStore] insert_hierarchy 失败，已回滚: {e}")
                raise MemoryStoreException(str(e))

        stats = {
            "conversations": 1,
            "topics": len(topic_rows),
            "memories": len(memory_rows),
        }
        logger.info(
            f"[MysqlMemoryStore] 三层入库完成: conversation="
            f"{conversation_item.conversation_id}, {stats}"
        )
        return stats

    # ------------------------------------------------------------------
    # 单层 upsert
    # ------------------------------------------------------------------
    def upsert_conversation(self, item: ConversationItem) -> None:
        with self._lock, self._conn() as conn:
            conn.cursor().execute(CONVERSATION_UPSERT_SQL, conversation_to_row(item))

    def upsert_topics(self, items: Sequence[TopicItem]) -> int:
        if not items:
            return 0
        rows = [topic_to_row(t) for t in items]
        with self._lock, self._conn() as conn:
            conn.cursor().executemany(TOPIC_UPSERT_SQL, rows)
        return len(rows)

    def upsert_topic(self, item: TopicItem) -> None:
        self.upsert_topics([item])

    def upsert_memories(self, items: Sequence[MemoryItem]) -> int:
        if not items:
            return 0
        rows = [memory_to_row(m) for m in items]
        with self._lock, self._conn() as conn:
            conn.cursor().executemany(MEMORY_UPSERT_SQL, rows)
        return len(rows)

    def upsert_memory(self, item: MemoryItem) -> None:
        self.upsert_memories([item])

    # ==================================================================
    # 阶段二：合并 —— sleep mode 下的记忆融合（仅 memory layer）
    # ==================================================================
    def merge_memories(
        self,
        user_id: str,
        source_memory_ids: List[str],
        merged_content: str,
        memory_type: str,
        memory_category: str,
        importance: float = 0.5,
        confidence: float = 0.5,
        metadata: Optional[Dict] = None,
    ) -> MemoryItem:
        """将多条旧记忆合并为一条新记忆。

        流程：
        1. 查询所有源记忆，聚合 source_topic_ids 与 source_topic_cites；
        2. 将源记忆置为 archived；
        3. 生成新记忆（derived_from_memory_ids 指向源记忆，derived_memory_count
           记录合并条数）并写入。
        """
        source_items = self.get_memories_by_ids(source_memory_ids)
        if not source_items:
            raise MemoryStoreException(f"合并失败: 未找到源记忆 {source_memory_ids}")

        all_topic_ids: List[str] = []
        all_cites: Dict[str, List[int]] = {}
        seen_topics: set = set()

        for src in source_items:
            for tid in src.source_topic_ids:
                tid = str(tid)
                if tid not in seen_topics:
                    seen_topics.add(tid)
                    all_topic_ids.append(tid)
            for tid, indices in (src.source_topic_cites or {}).items():
                tid = str(tid)
                merged = set(all_cites.get(tid, []))
                merged.update(int(i) for i in indices)
                all_cites[tid] = sorted(merged)

        self.update_status(source_memory_ids, "archived")

        merged_item = MemoryItem(
            memory_id=str(generate_id()),
            user_id=user_id,
            memory_content=merged_content,
            memory_type=memory_type,
            memory_category=memory_category,
            status="active",
            created_at=_now_iso(),
            importance=importance,
            confidence=confidence,
            retrieval_count=0,
            last_retrieved_at="",
            source_topic_ids=all_topic_ids,
            source_topic_cites=all_cites,
            derived_from_memory_ids=[str(i) for i in source_memory_ids],
            derived_memory_count=len(source_memory_ids),
            metadata=metadata or {},
        )
        self.upsert_memory(merged_item)
        logger.info(
            f"[MysqlMemoryStore] 合并阶段: user={user_id}, "
            f"合并 {len(source_memory_ids)} 条 -> {merged_item.memory_id}"
        )
        return merged_item

    # ==================================================================
    # memory layer 查询 / 状态
    # ==================================================================
    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM memories WHERE memory_id=%s", (str(memory_id),))
            row = cur.fetchone()
        return row_to_memory_item(dict(row)) if row else None

    def get_memories_by_ids(self, memory_ids: Sequence[str]) -> List[MemoryItem]:
        if not memory_ids:
            return []
        placeholders = ",".join(["%s"] * len(memory_ids))
        sql = f"SELECT * FROM memories WHERE memory_id IN ({placeholders})"
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute(sql, tuple(str(i) for i in memory_ids))
            rows = cur.fetchall()
        return [row_to_memory_item(dict(r)) for r in rows]

    def list_user_memories(
        self,
        user_id: str,
        status: Optional[str] = "active",
        memory_type: Optional[str] = None,
        memory_category: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[MemoryItem]:
        sql = "SELECT * FROM memories WHERE user_id=%s"
        params: List[Any] = [user_id]
        if status:
            sql += " AND status=%s"
            params.append(status)
        if memory_type:
            sql += " AND memory_type=%s"
            params.append(memory_type)
        if memory_category:
            sql += " AND memory_category=%s"
            params.append(memory_category)
        sql += " ORDER BY created_at DESC"
        logger.debug(f"[MysqlMemoryStore] list_user_memories sql={sql}, params={params}")
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
        return [row_to_memory_item(dict(r)) for r in rows]

    def update_status(self, memory_ids: Sequence[str], status: str) -> int:
        if not memory_ids:
            return 0
        placeholders = ",".join(["%s"] * len(memory_ids))
        sql = f"UPDATE memories SET status=%s WHERE memory_id IN ({placeholders})"
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute(sql, (status, *[str(i) for i in memory_ids]))
            return cur.rowcount

    def bump_retrieval(self, memory_ids: Sequence[str]) -> int:
        if not memory_ids:
            return 0
        placeholders = ",".join(["%s"] * len(memory_ids))
        sql = (
            "UPDATE memories SET retrieval_count = retrieval_count + 1, "
            f"last_retrieved_at = %s WHERE memory_id IN ({placeholders})"
        )
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute(sql, (_now_iso(), *[str(i) for i in memory_ids]))
            return cur.rowcount

    def delete_memory(self, memory_id: str) -> int:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM memories WHERE memory_id=%s", (str(memory_id),))
            return cur.rowcount

    def list_user_ids(self) -> List[str]:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT user_id FROM memories")
            rows = cur.fetchall()
        return [str(r["user_id"]) for r in rows]

    # ==================================================================
    # topic / conversation 查询（合并、追溯时使用）
    # ==================================================================
    def get_topic(self, topic_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM topics WHERE topic_id=%s", (str(topic_id),))
            row = cur.fetchone()
        return _decode_topic_row(dict(row)) if row else None

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM conversations WHERE conversation_id=%s",
                (str(conversation_id),),
            )
            row = cur.fetchone()
        return _decode_conversation_row(dict(row)) if row else None

    def get_topics_by_ids(self, topic_ids: Sequence[str]) -> List[Dict[str, Any]]:
        if not topic_ids:
            return []
        placeholders = ",".join(["%s"] * len(topic_ids))
        sql = f"SELECT * FROM topics WHERE topic_id IN ({placeholders})"
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute(sql, tuple(str(i) for i in topic_ids))
            rows = cur.fetchall()
        return [_decode_topic_row(dict(r)) for r in rows]

    def get_conversations_by_ids(
        self, conversation_ids: Sequence[str]
    ) -> List[Dict[str, Any]]:
        if not conversation_ids:
            return []
        placeholders = ",".join(["%s"] * len(conversation_ids))
        sql = f"SELECT * FROM conversations WHERE conversation_id IN ({placeholders})"
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute(sql, tuple(str(i) for i in conversation_ids))
            rows = cur.fetchall()
        return [_decode_conversation_row(dict(r)) for r in rows]

    # ==================================================================
    # 溯源：memory -> topic(s) -> conversation(s)
    # ==================================================================
    def get_memory_provenance(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """顺藤摸瓜：由 memory_id 一次取回记忆层、主题层、原始会话层的完整溯源。

        关联路径：
            memory.source_topic_ids        -> topics.topic_id
            topic.source_conversation_ids  -> conversations.conversation_id

        返回:
            {
                "memory": MemoryItem,
                "topics": [topic_row, ...],          # topic_row 内含 topic_messages（带 [n] 索引原文）
                "conversations": [conversation_row, ...],
                "cites": {topic_id: [msg_index, ...]},  # 精确到主题内的消息编号
                "cited_messages": {                  # 已把 cite 的 [n] 还原成具体消息
                    topic_id: [{"index": n, "speaker": ..., "content": ...}, ...],
                },
            }
        未找到 memory 时返回 None。
        """
        memory = self.get_memory(memory_id)
        if memory is None:
            return None

        topics = self.get_topics_by_ids(memory.source_topic_ids)
        topic_by_id = {str(t.get("topic_id")): t for t in topics}

        conversation_ids: List[str] = []
        seen: set = set()
        for topic in topics:
            for cid in topic.get("source_conversation_ids") or []:
                cid = str(cid)
                if cid not in seen:
                    seen.add(cid)
                    conversation_ids.append(cid)

        conversations = self.get_conversations_by_ids(conversation_ids)

        cites = memory.source_topic_cites or {}
        cited_messages: Dict[str, List[Dict[str, Any]]] = {}
        for topic_id, indices in cites.items():
            topic = topic_by_id.get(str(topic_id))
            if not topic:
                continue
            idx_to_msg = {
                int(m.get("index")): m
                for m in (topic.get("topic_messages") or [])
                if m.get("index") is not None
            }
            cited_messages[str(topic_id)] = [
                idx_to_msg[i] for i in indices if i in idx_to_msg
            ]

        return {
            "memory": memory,
            "topics": topics,
            "conversations": conversations,
            "cites": cites,
            "cited_messages": cited_messages,
        }

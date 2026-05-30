"""src.db 统一工厂：按 config.mysql.USE_SQLITE 路由到 MySQL 或 SQLite 实现。

外层只 import 这里：

>>> from src.db import create_memory_store
>>> store = create_memory_store()
>>> store.list_user_memories("user_001")

具体后端类型由 .env 中的 `mysql_use_sqlite` 决定。
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

from config.setting import Config, MysqlConfig
from src.db.base import (
    MemoryStoreClient,
    MemoryStoreException,
    memory_item_to_dict,
    memory_item_to_row,
    row_to_memory_item,
)


def create_memory_store(cfg: Optional[MysqlConfig] = None) -> MemoryStoreClient:
    """根据配置返回 MySQL 或 SQLite 的 MemoryItem 存储客户端。"""
    cfg = cfg or Config.mysql
    if cfg.USE_SQLITE:
        from src.db.sqllite.sqllite_memory_client import SqliteMemoryStore

        return SqliteMemoryStore(cfg=cfg)

    from src.db.mysql.mysql_memory_store import MysqlMemoryStore

    return MysqlMemoryStore(cfg=cfg)


__all__ = [
    "create_memory_store",
    "MemoryStoreClient",
    "MemoryStoreException",
    "memory_item_to_dict",
    "memory_item_to_row",
    "row_to_memory_item",
]

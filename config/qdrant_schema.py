#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Qdrant collection schema 常量（与 MemoryItem 存储约定一一对应）。

与 ``setting.py`` 区分:
    - ``setting.py``: 运行时可调参数, 走 .env (URL / API_KEY / DENSE_DIM / DISTANCE / ...)
    - ``qdrant_schema.py``: 业务 schema 约定 (vector 命名 / payload 索引), 改动 = 数据迁移

schema_type 用字符串表达, 让 config 层 **不依赖** ``qdrant_client.http.models``,
client 端再 resolve 回 ``models.PayloadSchemaType`` 枚举.
"""

from __future__ import annotations

from typing import Final, Tuple


# ---------------------------------------------------------------------- #
# Named vectors
# ---------------------------------------------------------------------- #
DENSE_VECTOR_NAME: Final[str] = "dense_embedding"
SPARSE_VECTOR_NAME: Final[str] = "sparse_embedding"


# ---------------------------------------------------------------------- #
# Payload schema types (字符串别名, 与 qdrant PayloadSchemaType 成员小写一致)
# ---------------------------------------------------------------------- #
KEYWORD: Final[str] = "keyword"
DATETIME: Final[str] = "datetime"
FLOAT: Final[str] = "float"
INTEGER: Final[str] = "integer"
BOOL: Final[str] = "bool"
TEXT: Final[str] = "text"

# 所有合法的 schema_type 字符串, 供 client 端 resolve / 校验
SUPPORTED_PAYLOAD_SCHEMA_TYPES: Final[Tuple[str, ...]] = (
    KEYWORD, DATETIME, FLOAT, INTEGER, BOOL, TEXT,
)


# ---------------------------------------------------------------------- #
# Payload index 定义
# (field_name, schema_type)
#
# 调整原则:
#   - 高基数 + 精确匹配字段 -> KEYWORD
#   - 时间字段 -> DATETIME (支持范围)
#   - 评分类数值 -> FLOAT (支持范围)
# ---------------------------------------------------------------------- #
PAYLOAD_INDEXES: Final[Tuple[Tuple[str, str], ...]] = (
    ("user_id", KEYWORD),
    ("memory_id", KEYWORD),
    ("memory_type", KEYWORD),
    ("memory_category", KEYWORD),
    ("status", KEYWORD),
    ("content_hash", KEYWORD),
    ("created_at", DATETIME),
    ("importance", FLOAT),
    ("confidence", FLOAT),
)


__all__ = [
    "DENSE_VECTOR_NAME",
    "SPARSE_VECTOR_NAME",
    "KEYWORD",
    "DATETIME",
    "FLOAT",
    "INTEGER",
    "BOOL",
    "TEXT",
    "SUPPORTED_PAYLOAD_SCHEMA_TYPES",
    "PAYLOAD_INDEXES",
]

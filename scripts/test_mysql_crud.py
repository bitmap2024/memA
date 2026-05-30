#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""临时脚本：测试 MySQL 记忆层 建表 / 插入 / 更新 / 查询（新 schema）。"""

import os
import sys
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("mysql_use_sqlite", "false")

import pymysql
from pymysql.cursors import DictCursor

from config.releatiion_schema import MemoryItem
from config.setting import Config
from src.db.mysql.mysql_memory_store import MysqlMemoryStore
from src.utils.snow_id import generate_id


def main() -> None:
    cfg = Config.mysql
    print("=== MySQL 连接配置 ===")
    print(f"  HOST: {cfg.HOST}:{cfg.PORT}")
    print(f"  USER: {cfg.USER}")
    print(f"  DATABASE: {cfg.DATABASE}")

    # ---------- 1. 建表 ----------
    print("\n=== 1. 连接并自动建表 (ensure_schema) ===")
    store = MysqlMemoryStore(cfg=cfg)
    print(f"  后端: {store.backend}")

    conn = pymysql.connect(
        host=cfg.HOST,
        port=cfg.PORT,
        user=cfg.USER,
        password=cfg.PASSWORD,
        database=cfg.DATABASE,
        charset="utf8mb4",
        cursorclass=DictCursor,
        autocommit=True,
    )
    cur = conn.cursor()
    cur.execute("SHOW TABLES")
    tables = [list(r.values())[0] for r in cur.fetchall()]
    print(f"  已有表: {tables}")

    # ---------- 2. 插入 (项目 API) ----------
    print("\n=== 2. 插入记忆 (store.upsert_memory) ===")
    mem_id = str(generate_id())
    user_id = "test_user_001"
    content = "用户喜欢喝拿铁咖啡"
    item = MemoryItem(
        memory_id=mem_id,
        user_id=user_id,
        memory_content=content,
        memory_type="profile",
        memory_category="preference",
        status="active",
        importance=0.8,
        confidence=0.9,
        metadata={"source": "manual_test"},
    )
    store.upsert_memory(item)
    print(f"  已插入 memory_id={mem_id}")

    # ---------- 3. 查询 (原生 SQL) ----------
    print("\n=== 3. 查询 (SELECT) ===")
    cur.execute(
        "SELECT memory_id, memory_content, importance, status "
        "FROM memories WHERE memory_id=%s",
        (mem_id,),
    )
    row = cur.fetchone()
    print(f"  content={row['memory_content']}, importance={row['importance']}")

    # ---------- 4. 更新 (项目 API: upsert 同 memory_id) ----------
    print("\n=== 4. 更新记忆 (upsert 同 id -> ON DUPLICATE KEY UPDATE) ===")
    item.memory_content = "用户喜欢喝美式咖啡，不加糖"
    item.importance = 0.95
    store.upsert_memory(item)
    cur.execute(
        "SELECT memory_content, importance FROM memories WHERE memory_id=%s",
        (mem_id,),
    )
    row = cur.fetchone()
    print(f"  更新后 content={row['memory_content']}, importance={row['importance']}")

    # ---------- 5. 更新状态 ----------
    print("\n=== 5. 更新状态 (store.update_status) ===")
    n = store.update_status([mem_id], "archived")
    cur.execute("SELECT status FROM memories WHERE memory_id=%s", (mem_id,))
    print(f"  影响行数={n}, status={cur.fetchone()['status']}")

    # ---------- 6. 清理 ----------
    print("\n=== 6. 清理测试数据 ===")
    store.delete_memory(mem_id)
    cur.execute("SELECT COUNT(*) AS cnt FROM memories WHERE memory_id=%s", (mem_id,))
    print(f"  剩余行数: {cur.fetchone()['cnt']}")

    conn.close()
    print("\n=== 全部测试通过 ===")


if __name__ == "__main__":
    main()

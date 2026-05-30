#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""UpdateMemoryService：sleep mode 合并 + category.md 文档生成。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger

from src.db import MemoryStoreClient, create_memory_store
from src.update_mem.category_doc_builder import CategoryDocBuilder
from src.update_mem.sleep_mode_update import SleepModeUpdater


class UpdateMemoryService:
    def __init__(
        self,
        sleep_updater: Optional[SleepModeUpdater] = None,
        doc_builder: Optional[CategoryDocBuilder] = None,
        store: Optional[MemoryStoreClient] = None,
    ) -> None:
        self.store = store or create_memory_store()
        self.sleep_updater = sleep_updater or SleepModeUpdater(store=self.store)
        self.doc_builder = doc_builder or CategoryDocBuilder(store=self.store)

    def update_user(
        self,
        user_id: str,
        max_workers: int = 4,
        enable_merge: bool = True,
        enable_doc: bool = True,
    ) -> Dict[str, Any]:
        report: Dict[str, Any] = {"user_id": user_id, "merge": None, "doc": None}
        if enable_merge:
            report["merge"] = self.sleep_updater.run(user_id=user_id, max_workers=max_workers)
        if enable_doc:
            report["doc"] = self.doc_builder.build_for_user(user_id=user_id, max_workers=max_workers)
        return report

    def update_all_users(
        self,
        max_workers: int = 4,
        enable_merge: bool = True,
        enable_doc: bool = True,
    ) -> List[Dict[str, Any]]:
        user_ids = self.store.list_user_ids()
        logger.info(f"[UpdateMemoryService] 共 {len(user_ids)} 个用户待更新")
        out: List[Dict[str, Any]] = []
        for uid in user_ids:
            try:
                out.append(
                    self.update_user(
                        user_id=uid,
                        max_workers=max_workers,
                        enable_merge=enable_merge,
                        enable_doc=enable_doc,
                    )
                )
            except Exception as e:
                logger.warning(f"[UpdateMemoryService] 用户 {uid} 更新失败: {e}")
                out.append({"user_id": uid, "error": str(e)})
        return out

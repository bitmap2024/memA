#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""UpdateMemoryService - offline memory update/merge service."""

from __future__ import annotations

from typing import Any, Dict, Optional

from loguru import logger

from config.config import Config
from update_mem.offline_memory_update import OfflineMemoryUpdate


class UpdateMemoryService:
    """Argo-facing orchestration for global memory update jobs."""

    def __init__(self, config: Config = None, updater: Optional[OfflineMemoryUpdate] = None):
        self.config = config or Config
        self.store_manage = updater or OfflineMemoryUpdate(config=self.config)

    def offline_update_memories(
        self,
        child_id: str = None,
        agent_id: str = None,
        score_threshold: float = 0.9,
        max_workers: int = 5,
        enable_merge: bool = True,
        top_k: int = 20,
        keep_top_n: int = 10,
    ) -> Dict[str, Any]:
        logger.info("=" * 50)
        logger.info("开始离线记忆更新任务")
        logger.info(
            f"参数: child_id={child_id}, agent_id={agent_id}, threshold={score_threshold}, "
            f"top_k={top_k}, keep_top_n={keep_top_n}, max_workers={max_workers}, enable_merge={enable_merge}"
        )
        logger.info("=" * 50)

        result: Dict[str, Any] = {
            "queue_constructed": False,
            "merge": {"clusters": 0, "merged": 0, "deleted": 0, "failed": 0},
            "updated": True,
        }

        self.store_manage.construct_update_queue_all_entries(
            child_id=child_id,
            agent_id=agent_id,
            top_k=top_k,
            keep_top_n=keep_top_n,
            max_workers=max_workers,
        )
        result["queue_constructed"] = True

        if enable_merge:
            result["merge"] = self.store_manage.merge_similar_memories(
                child_id=child_id,
                agent_id=agent_id,
                score_threshold=score_threshold,
                max_workers=max_workers,
            )

        self.store_manage.offline_update_all_entries(
            child_id=child_id,
            agent_id=agent_id,
            score_threshold=score_threshold,
            max_workers=max_workers,
        )

        logger.info(f"离线记忆更新任务完成: {result}")
        return result

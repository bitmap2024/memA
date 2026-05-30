#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger

from src.extract_mem.memory_extract_pipeline import MemoryExtractPipeline


class ExtractMemoryService:
    def __init__(self, pipeline: Optional[MemoryExtractPipeline] = None) -> None:
        self.pipeline = pipeline or MemoryExtractPipeline()

    def extract_one(
        self,
        user_id: str,
        conversation: List[Dict[str, str]],
        conversation_date_time: Optional[str] = None,
        session_id: Optional[str] = None,
        compress_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        history = {
            "user_id": user_id,
            "session_id": session_id,
            "conversation_date_time": conversation_date_time,
            "conversation": conversation,
        }
        return self.pipeline.extract_pipeline(history=history, compress_rate=compress_rate)

    def extract_batch(self, histories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for history in histories:
            try:
                results.append(self.pipeline.extract_pipeline(history=history))
            except Exception as e:
                logger.warning(f"[ExtractMemoryService] 抽取失败: {e}")
                results.append({"error": str(e), "history": history.get("user_id")})
        return results

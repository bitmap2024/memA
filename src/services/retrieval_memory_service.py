#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RetrievalMemoryService：对外提供混合检索接口。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.db import MemoryStoreClient, create_memory_store
from src.retrieval_mem.hybrid_retrieval import HybridRetrieval


class RetrievalMemoryService:
    def __init__(
        self,
        hybrid: Optional[HybridRetrieval] = None,
        store: Optional[MemoryStoreClient] = None,
    ) -> None:
        self.store = store or create_memory_store()
        self.hybrid = hybrid or HybridRetrieval(store=self.store)

    # ------------------------------------------------------------------
    # 在线 query 接口
    # ------------------------------------------------------------------
    def retrieve(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
        memory_type: Optional[str] = None,
        memory_category: Optional[str] = None,
        use_bge_rerank: bool = True,
        use_llm_rerank: bool = False,
        use_mmr: bool = True,
    ) -> List[Dict[str, Any]]:
        return self.hybrid.retrieve(
            user_id=user_id,
            query=query,
            top_k=top_k,
            memory_type=memory_type,
            memory_category=memory_category,
            use_bge_rerank=use_bge_rerank,
            use_llm_rerank=use_llm_rerank,
            use_mmr=use_mmr,
        )

    def get_relate_memory_contents(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
    ) -> List[str]:
        results = self.retrieve(user_id=user_id, query=query, top_k=top_k)
        seen = set()
        contents: List[str] = []
        for item in results:
            content = str(item.get("memory_content") or "")
            if not content or content in seen:
                continue
            seen.add(content)
            contents.append(content)
        return contents

    # ------------------------------------------------------------------
    # 离线 / 直接读
    # ------------------------------------------------------------------
    def get_all_memories(self, user_id: str, status: str = "active") -> List[Dict[str, Any]]:
        items = self.store.list_user_memories(user_id=user_id, status=status)
        return [
            {
                "memory_id": item.memory_id,
                "memory_content": item.memory_content,
                "memory_type": item.memory_type,
                "memory_category": item.memory_category,
                "importance": item.importance,
                "confidence": item.confidence,
                "status": item.status,
                "created_at": item.created_at,
                "source_topic_ids": item.source_topic_ids or [],
                "derived_from_memory_ids": item.derived_from_memory_ids or [],
            }
            for item in items
        ]

    def get_user_memory_by_type(
        self,
        user_id: str,
        memory_type: str,
        memory_category: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        items = self.store.list_user_memories(
            user_id=user_id,
            memory_type=memory_type,
            memory_category=memory_category,
            limit=limit,
        )
        return [
            {
                "memory_id": item.memory_id,
                "memory_content": item.memory_content,
                "memory_type": item.memory_type,
                "memory_category": item.memory_category,
                "importance": item.importance,
                "confidence": item.confidence,
                "created_at": item.created_at,
            }
            for item in items
        ]

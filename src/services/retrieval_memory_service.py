#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RetrievalMemoryService - online memory retrieval service."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List
from zoneinfo import ZoneInfo

import Levenshtein
from loguru import logger

from api.qdrant_api import QdrantMemoryClient
from config.config import Config
from extract_mem.memory_extract_pipeline import MEMORY_TYPE_MAPPING
from retrieval_mem.memory_retrieval import MemoryRetrieval
from utils.utils import utc_convert_beijing


class RetrievalMemoryService:
    """Online query and read API for memories."""

    memory_type_mapping = MEMORY_TYPE_MAPPING

    def __init__(self, config: Config = None):
        self.config = config or Config
        self.mem_search = MemoryRetrieval(config=self.config)
        self.vector_store = QdrantMemoryClient(
            collection_name=self.config.qdrant.COLLECTION,
            vector_size=self.config.qdrant.VECTOR_SIZE,
            distance=self.config.qdrant.DISTANCE,
        )

    def get_relate_memory(self, child_id: str, agent_id: str, query: str, top_k: int = 10) -> List[str]:
        results = self.mem_search.retrieve(
            query=query,
            child_id=child_id,
            agent_id=agent_id,
            top_k=top_k,
        )
        return self.memory_format(results)

    def get_memory_with_date_limit(
        self,
        child_id: str,
        agent_id: str = None,
        days: int = 30,
        limit: int = None,
    ) -> List[Dict]:
        current_utc = datetime.now(ZoneInfo("UTC"))
        days_ago = (current_utc - timedelta(days=days)).isoformat()

        results = self.vector_store.scroll_all(
            payload_filter=QdrantMemoryClient.build_filter(
                child_id=child_id,
                agent_id=agent_id,
                updated_at_gte=days_ago,
                updated_at_lte=current_utc.isoformat(),
            ),
            with_vectors=False,
        )
        results = QdrantMemoryClient.sort_records(results, "updated_at", reverse=True)
        results = self.online_deduplicate_memories(results, max_distance=1)
        return results[:limit] if limit else results

    def get_all_memory(self, child_id: str, agent_id: str = None) -> List[Dict]:
        all_memories = self.vector_store.scroll_all(
            payload_filter=QdrantMemoryClient.build_filter(
                child_id=child_id,
                agent_id=agent_id,
            ),
            limit=200,
            with_vectors=False,
        )

        memory_type_e2z = {v: k for k, v in self.memory_type_mapping.items()}
        format_all_memories = defaultdict(list)

        for memory in all_memories:
            memory_content = memory.get("memory_content", "")
            memory_type = memory.get("memory_type", "")
            updated_at = memory.get("updated_at", "")
            try:
                updated_at = utc_convert_beijing(updated_at)
            except Exception:
                pass

            key = (memory_type, memory_content)
            format_all_memories[key].append(
                {
                    "memory_content": memory_content,
                    "memory_type": memory_type_e2z.get(memory_type, memory_type),
                    "updated_at": updated_at,
                }
            )

        result = []
        for memories in format_all_memories.values():
            result.extend(memories)
        return result

    def memory_format(self, memories: List[Dict]) -> List[str]:
        try:
            deduplicated = self.online_deduplicate_memories(memories, max_distance=1)
        except Exception as e:
            logger.warning(f"去重失败: {e}")
            deduplicated = memories

        seen = set()
        memory_strings = []
        for memory_item in deduplicated:
            memory_content = memory_item.get("memory_content", "")
            if memory_content and memory_content not in seen:
                seen.add(memory_content)
                memory_strings.append(memory_content)
        return memory_strings

    def online_deduplicate_memories(self, memories: List[Dict], max_distance: int = 1) -> List[Dict]:
        if not memories:
            return []

        grouped = defaultdict(list)
        for memory in memories:
            grouped[memory.get("memory_type", "other")].append(memory)

        deduplicated = []
        for memory_list in grouped.values():
            unique = []
            for memory in memory_list:
                content = memory.get("memory_content", "")
                is_duplicate = False
                for unique_memory in unique:
                    unique_content = unique_memory.get("memory_content", "")
                    if Levenshtein.distance(content, unique_content) <= max_distance:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique.append(memory)
            deduplicated.extend(unique)
        return deduplicated

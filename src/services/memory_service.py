#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MemoryService compatibility facade.

The architecture is split into:
- ExtractMemoryService: offline extract and direct insert.
- UpdateMemoryService: offline global update/merge.
- RetrievalMemoryService: online query and read APIs.
"""

from __future__ import annotations

from typing import Dict, List

from config.config import Config
from extract_mem.memory_extract_pipeline import MEMORY_TYPE_MAPPING
from services.extract_memory_service import ExtractMemoryService
from services.retrieval_memory_service import RetrievalMemoryService
from services.update_memory_service import UpdateMemoryService


class MemoryService(ExtractMemoryService, UpdateMemoryService, RetrievalMemoryService):
    """Backward-compatible facade for existing legacy callers."""

    memory_type_mapping = MEMORY_TYPE_MAPPING

    def __init__(self, config: Config = None):
        self.config = config or Config
        self.extract_service = ExtractMemoryService(config=self.config)
        self.update_service = UpdateMemoryService(config=self.config)
        self.retrieval_service = RetrievalMemoryService(config=self.config)

    def offline_extract_memories(self, *args, **kwargs):
        return self.extract_service.offline_extract_memories(*args, **kwargs)

    def offline_process_memories(self, *args, **kwargs):
        return self.extract_service.offline_extract_memories(*args, **kwargs)

    def process_single_user_sync(self, *args, **kwargs):
        return self.extract_service.process_single_user_sync(*args, **kwargs)

    def _process_single_user(self, *args, **kwargs):
        return self.extract_service._process_single_user(*args, **kwargs)

    def _get_user_ids(self, *args, **kwargs):
        return self.extract_service._get_user_ids(*args, **kwargs)

    def _fetch_user_history(self, *args, **kwargs):
        return self.extract_service._fetch_user_history(*args, **kwargs)

    def _step1_normalize_messages(self, *args, **kwargs):
        return self.extract_service._step1_normalize_messages(*args, **kwargs)

    def _step2_compress_text(self, *args, **kwargs):
        return self.extract_service._step2_compress_text(*args, **kwargs)

    def _step3_chunk_messages(self, *args, **kwargs):
        return self.extract_service._step3_chunk_messages(*args, **kwargs)

    def _step4_extract_memories(self, *args, **kwargs):
        return self.extract_service._step4_extract_memories(*args, **kwargs)

    def _step5_store_memories(self, *args, **kwargs):
        return self.extract_service._step5_store_memories(*args, **kwargs)

    def _messages_to_text(self, *args, **kwargs):
        return self.extract_service._messages_to_text(*args, **kwargs)

    def _parse_memory_response(self, *args, **kwargs):
        return self.extract_service._parse_memory_response(*args, **kwargs)

    def _generate_memory_id(self, *args, **kwargs):
        return self.extract_service._generate_memory_id(*args, **kwargs)

    def offline_update_memories(self, *args, **kwargs):
        return self.update_service.offline_update_memories(*args, **kwargs)

    def get_relate_memory(self, child_id: str, agent_id: str, query: str, top_k: int = 10) -> List[str]:
        return self.retrieval_service.get_relate_memory(child_id, agent_id, query, top_k)

    def get_memory_with_date_limit(self, *args, **kwargs) -> List[Dict]:
        return self.retrieval_service.get_memory_with_date_limit(*args, **kwargs)

    def get_all_memory(self, *args, **kwargs) -> List[Dict]:
        return self.retrieval_service.get_all_memory(*args, **kwargs)

    def memory_format(self, *args, **kwargs) -> List[str]:
        return self.retrieval_service.memory_format(*args, **kwargs)

    def online_deduplicate_memories(self, *args, **kwargs) -> List[Dict]:
        return self.retrieval_service.online_deduplicate_memories(*args, **kwargs)


def create_memory_service(config: Config = None) -> MemoryService:
    return MemoryService(config=config)


if __name__ == "__main__":
    service = MemoryService(config=Config)
    print("MemoryService facade loaded")

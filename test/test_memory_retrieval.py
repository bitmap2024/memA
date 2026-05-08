#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试记忆检索模块。
"""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMemoryRetrieval:
    def test_retrieve_basic(self):
        with patch("retrieval_mem.memory_retrieval.EmbeddingClientPool") as mock_pool, \
             patch("retrieval_mem.memory_retrieval.QdrantMemoryClient") as mock_store, \
             patch("retrieval_mem.memory_retrieval.MMRSearch"):
            mock_client = MagicMock()
            mock_client.get_embeddings.return_value = np.random.rand(1, 768)
            mock_pool.return_value.get_client.return_value = mock_client
            mock_store.return_value.search.return_value = [
                {
                    "_id": "1",
                    "_score": 0.91,
                    "memory_id": "1",
                    "memory_content": "记忆1",
                    "memory_type": "preference",
                    "updated_at": "2025-02-15T10:00:00Z",
                    "vector": np.random.rand(768).tolist(),
                }
            ]

            from retrieval_mem.memory_retrieval import MemoryRetrieval

            retriever = MemoryRetrieval(collection_name="test_collection")
            results = retriever.retrieve(query="测试查询", top_k=10, use_mmr=False)

            assert len(results) == 1
            assert results[0]["memory_content"] == "记忆1"
            mock_store.return_value.search.assert_called_once()

    def test_retrieve_by_type(self):
        with patch("retrieval_mem.memory_retrieval.EmbeddingClientPool") as mock_pool, \
             patch("retrieval_mem.memory_retrieval.QdrantMemoryClient") as mock_store, \
             patch("retrieval_mem.memory_retrieval.MMRSearch"):
            mock_pool.return_value.get_client.return_value = MagicMock()
            mock_store.return_value.scroll_all.return_value = [
                {"memory_content": "记忆1", "memory_type": "preference", "updated_at": "2025-02-15T11:00:00Z"},
                {"memory_content": "记忆2", "memory_type": "preference", "updated_at": "2025-02-15T10:00:00Z"},
            ]

            from retrieval_mem.memory_retrieval import MemoryRetrieval

            retriever = MemoryRetrieval(collection_name="test_collection")
            results = retriever.retrieve_by_type(child_id="child", memory_type="preference", limit=1)

            assert len(results) == 1
            assert results[0]["memory_content"] == "记忆1"

    def test_retrieve_recent(self):
        with patch("retrieval_mem.memory_retrieval.EmbeddingClientPool") as mock_pool, \
             patch("retrieval_mem.memory_retrieval.QdrantMemoryClient") as mock_store, \
             patch("retrieval_mem.memory_retrieval.MMRSearch"):
            mock_pool.return_value.get_client.return_value = MagicMock()
            mock_store.return_value.scroll_all.return_value = [
                {"memory_content": "最新记忆", "memory_type": "preference", "updated_at": "2025-02-16T11:00:00Z"},
                {"memory_content": "旧记忆", "memory_type": "preference", "updated_at": "2025-02-15T10:00:00Z"},
            ]

            from retrieval_mem.memory_retrieval import MemoryRetrieval

            retriever = MemoryRetrieval(collection_name="test_collection")
            results = retriever.retrieve_recent(child_id="child", days=7, limit=1)

            assert len(results) == 1
            assert results[0]["memory_content"] == "最新记忆"


class TestMMRSearch:
    def test_mmr_basic(self):
        from retrieval_mem.mmr_search import MMRSearch

        mmr = MMRSearch(lambda_param=0.5)
        query_embedding = np.random.rand(768)
        candidate_embeddings = np.random.rand(10, 768)
        candidate_items = [{"id": i, "content": f"内容{i}"} for i in range(10)]

        results = mmr.search(
            query_embedding=query_embedding,
            candidate_embeddings=candidate_embeddings,
            candidate_items=candidate_items,
            top_k=5,
        )

        assert len(results) <= 5


class TestMemoryServiceRetrieval:
    def test_get_relate_memory(self):
        from services.retrieval_memory_service import RetrievalMemoryService

        with patch.object(RetrievalMemoryService, "__init__", lambda x, **kwargs: None):
            service = RetrievalMemoryService.__new__(RetrievalMemoryService)
            service.mem_search = MagicMock()
            service.mem_search.retrieve.return_value = [
                {"memory_content": "记忆1", "child_id": "test_child", "score": 0.9},
                {"memory_content": "记忆2", "child_id": "test_child", "score": 0.8},
            ]
            service.memory_format = MagicMock(return_value=["记忆1", "记忆2"])

            results = service.get_relate_memory(
                child_id="test_child",
                agent_id="test_agent",
                query="测试查询",
            )

            assert results == ["记忆1", "记忆2"]
            service.mem_search.retrieve.assert_called_once()

    def test_get_all_memory(self):
        from services.retrieval_memory_service import RetrievalMemoryService

        with patch.object(RetrievalMemoryService, "__init__", lambda x, **kwargs: None):
            service = RetrievalMemoryService.__new__(RetrievalMemoryService)
            service.vector_store = MagicMock()
            service.vector_store.scroll_all.return_value = [
                {"memory_content": "记忆1", "memory_type": "preference", "updated_at": "2025-02-15T10:00:00Z"},
                {"memory_content": "记忆2", "memory_type": "factual", "updated_at": "2025-02-15T11:00:00Z"},
            ]
            service.memory_type_mapping = {"偏好": "preference", "事实": "factual"}

            results = service.get_all_memory(child_id="test_child")

            assert len(results) == 2
            service.vector_store.scroll_all.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

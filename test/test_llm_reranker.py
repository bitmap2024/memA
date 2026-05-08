#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for LLM memory reranker."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


class FakeLLM:
    def __init__(self, response):
        self.response = response

    def chat(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def test_llm_reranker_reorders_memory_dicts():
    from reranker.llm_reranker import LLMReranker

    llm = FakeLLM(
        """
        {
          "ranked": [
            {"index": 1, "score": 0.95, "reason": "更匹配奥特曼偏好"},
            {"index": 0, "score": 0.4, "reason": "弱相关"}
          ]
        }
        """
    )
    memories = [
        {"memory_id": "1", "memory_content": "孩子喜欢蓝色", "score": 0.9},
        {"memory_id": "2", "memory_content": "孩子非常喜欢奥特曼", "score": 0.8},
    ]

    results = LLMReranker(llm=llm).rerank("孩子喜欢什么动画角色", memories, top_k=2)

    assert [item["memory_id"] for item in results] == ["2", "1"]
    assert results[0]["llm_rank"] == 1
    assert results[0]["llm_score"] == 0.95
    assert "llm_rank" not in memories[0]


def test_llm_reranker_falls_back_on_invalid_response():
    from reranker.llm_reranker import LLMReranker

    memories = [
        {"memory_id": "1", "memory_content": "记忆1"},
        {"memory_id": "2", "memory_content": "记忆2"},
    ]

    results = LLMReranker(llm=FakeLLM("not json")).rerank("测试", memories, top_k=1)

    assert results == [memories[0]]

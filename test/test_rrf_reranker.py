#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for RRF memory reranker."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


def test_rrf_reranker_fuses_ranked_memory_lists():
    from reranker.rrf_reranker import RRFReranker

    vector_results = [
        {"memory_id": "a", "memory_content": "孩子喜欢蓝色"},
        {"memory_id": "b", "memory_content": "孩子喜欢奥特曼"},
        {"memory_id": "c", "memory_content": "孩子喜欢骑车"},
    ]
    keyword_results = [
        {"memory_id": "b", "memory_content": "孩子喜欢奥特曼"},
        {"memory_id": "d", "memory_content": "孩子最近在看动画片"},
        {"memory_id": "a", "memory_content": "孩子喜欢蓝色"},
    ]

    results = RRFReranker(rank_constant=60).rerank(
        [vector_results, keyword_results],
        top_k=3,
    )

    assert [item["memory_id"] for item in results] == ["b", "a", "d"]
    assert results[0]["rrf_rank"] == 1
    assert len(results[0]["rrf_sources"]) == 2
    assert "rrf_rank" not in vector_results[0]


def test_rrf_reranker_supports_weights():
    from reranker.rrf_reranker import rrf_rerank

    results = rrf_rerank(
        ranked_lists=[
            [{"memory_id": "a"}, {"memory_id": "b"}],
            [{"memory_id": "b"}, {"memory_id": "a"}],
        ],
        weights=[1.0, 3.0],
        top_k=1,
    )

    assert results[0]["memory_id"] == "b"

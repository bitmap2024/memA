#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for memory extraction primitives."""

import json
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMemoryExtractor:
    def test_extract_basic(self):
        mock_response = json.dumps([
            {"content": "孩子非常喜欢奥特曼", "type": "偏好记忆"},
            {"content": "孩子学会了骑自行车", "type": "能力与发展"},
        ])

        with patch("extract_mem.memory_extractor.LLMApi") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm.chat.return_value = mock_response
            mock_llm_class.return_value = mock_llm

            from extract_mem.memory_extractor import LLmExtractor

            extractor = LLmExtractor()
            extractor.llm = mock_llm
            result = extractor.extract("用户: 我喜欢奥特曼")

            assert result == mock_response
            mock_llm.chat.assert_called_once()

    def test_extract_uses_prompt(self):
        with patch("extract_mem.memory_extractor.LLMApi") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm.chat.return_value = "[]"
            mock_llm_class.return_value = mock_llm

            from extract_mem.memory_extractor import LLmExtractor

            extractor = LLmExtractor()
            extractor.llm = mock_llm
            text = "用户: 我最喜欢蓝色"
            extractor.extract(text)

            assert text in str(mock_llm.chat.call_args)


class TestMemoryExtractionParsing:
    def test_parse_json_array(self):
        from extract_mem.memory_extract_pipeline import MemoryExtractPipeline

        pipeline = MemoryExtractPipeline.__new__(MemoryExtractPipeline)
        response = '[{"content": "记忆1", "type": "偏好"}, {"content": "记忆2", "type": "事实"}]'

        result = pipeline.parse_memory_response(response)

        assert len(result) == 2
        assert result[0]["content"] == "记忆1"

    def test_parse_json_object_with_memories(self):
        from extract_mem.memory_extract_pipeline import MemoryExtractPipeline

        pipeline = MemoryExtractPipeline.__new__(MemoryExtractPipeline)
        response = '{"memories": [{"content": "记忆1"}, {"content": "记忆2"}]}'

        result = pipeline.parse_memory_response(response)

        assert len(result) == 2

    def test_parse_string_memories(self):
        from extract_mem.memory_extract_pipeline import MemoryExtractPipeline

        pipeline = MemoryExtractPipeline.__new__(MemoryExtractPipeline)
        result = pipeline.parse_memory_response('["记忆1", "记忆2"]')

        assert result == [
            {"content": "记忆1", "type": "other"},
            {"content": "记忆2", "type": "other"},
        ]

    def test_parse_invalid_json(self):
        from extract_mem.memory_extract_pipeline import MemoryExtractPipeline

        pipeline = MemoryExtractPipeline.__new__(MemoryExtractPipeline)

        assert pipeline.parse_memory_response("这不是 JSON") == []

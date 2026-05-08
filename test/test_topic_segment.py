#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for topic segmentation."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


class CharTokenizer:
    def encode(self, text):
        return list(text)


def one_hot_embeddings(texts):
    return [
        [1.0 if index == col else 0.0 for col in range(len(texts))]
        for index in range(len(texts))
    ]


class TestTopicSegmenter:
    def test_short_messages_are_not_segmented(self):
        from extract_mem.topic_segment import TopicSegmenter

        segmenter = TopicSegmenter(
            embedding_fn=lambda texts: [[1.0, 1.0] for _ in texts],
            token_threshold=512,
            tokenizer=CharTokenizer(),
        )
        messages = [
            {"role": "user", "content": "我喜欢蓝色"},
            {"role": "assistant", "content": "蓝色很好看"},
        ]

        result = segmenter.segment_with_details(messages)

        assert result.segmented is False
        assert result.similarity_matrix == []
        assert segmenter.segment(messages) == [messages]

    def test_long_messages_are_segmented_by_adjacent_similarity(self):
        from extract_mem.topic_segment import TopicSegmenter

        embeddings = [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ]
        segmenter = TopicSegmenter(
            embedding_fn=lambda texts: embeddings,
            token_threshold=10,
            similarity_threshold=0.5,
            tokenizer=CharTokenizer(),
        )
        messages = [
            {"role": "user", "content": "我喜欢蓝色衣服"},
            {"role": "assistant", "content": "蓝色衣服很好看"},
            {"role": "user", "content": "我还想买红色外套"},
            {"role": "assistant", "content": "红色外套也不错"},
            {"role": "user", "content": "今天数学作业有点难"},
            {"role": "assistant", "content": "我们一起看题目"},
        ]

        result = segmenter.segment_with_details(messages)

        assert result.segmented is True
        assert len(result.segments) == 2
        assert result.segments[0].messages == messages[:4]
        assert result.segments[1].messages == messages[4:]
        assert len(result.similarity_matrix) == 3

    def test_user_assistant_content_records_are_supported(self):
        from extract_mem.topic_segment import TopicSegmenter

        segmenter = TopicSegmenter(
            embedding_fn=one_hot_embeddings,
            token_threshold=1,
            similarity_threshold=0.5,
            tokenizer=CharTokenizer(),
        )
        records = [
            {"user_content": "我想去公园", "assistant_content": "可以周末去"},
            {"user_content": "我要写作业", "assistant_content": "先写数学"},
        ]

        result = segmenter.segment_with_details(records)

        assert len(result.segments) == 2
        assert result.segments[0].records[0]["text"] == (
            "user: 我想去公园\nassistant: 可以周末去"
        )
        assert segmenter.segment(records) == [[records[0]], [records[1]]]

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Topic segmentation for LLM chat messages.

The segmenter keeps short conversations intact. When the conversation is over
the token budget, it folds each user/assistant turn into one record, embeds the
turn records, builds a cosine similarity matrix, and splits contiguous topics
where semantic similarity drops below the configured threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from loguru import logger


EmbeddingFn = Callable[[List[str]], Any]


@dataclass
class TopicSegment:
    """One contiguous topic segment."""

    topic_id: int
    messages: List[Dict[str, Any]]
    records: List[Dict[str, Any]]
    start_record_index: int
    end_record_index: int
    token_count: int


@dataclass
class TopicSegmentResult:
    """Detailed result for topic segmentation."""

    segments: List[TopicSegment]
    similarity_matrix: List[List[float]]
    total_tokens: int
    segmented: bool


class TopicSegmenter:
    """
    Split LLM messages into topic-level contiguous chunks.

    Args:
        embedding_client: Object that exposes ``get_embeddings(List[str])``.
        embedding_fn: Optional callable alternative to ``embedding_client``.
        token_threshold: Conversations no longer than this are not segmented.
        similarity_threshold: Start a new topic when adjacent turn similarity is
            below this value.
        tokenizer: Optional tokenizer with ``encode`` method.
    """

    USER_KEYS = ("user_content", "request_content", "RequestContent", "user")
    ASSISTANT_KEYS = (
        "assistant_content",
        "response_content",
        "ResponseContent",
        "assistant",
    )

    def __init__(
        self,
        embedding_client: Optional[Any] = None,
        embedding_fn: Optional[EmbeddingFn] = None,
        token_threshold: int = 512,
        similarity_threshold: float = 0.55,
        tokenizer: Optional[Any] = None,
    ):
        if embedding_client is None and embedding_fn is None:
            logger.warning("未配置 embedding_client/embedding_fn，长文本主题划分时将无法计算向量")

        self.embedding_client = embedding_client
        self.embedding_fn = embedding_fn
        self.token_threshold = max(1, int(token_threshold))
        self.similarity_threshold = float(similarity_threshold)
        self.tokenizer = tokenizer

    def segment(self, messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Return only message chunks, suitable for callers that do not need diagnostics."""

        return [segment.messages for segment in self.segment_with_details(messages).segments]

    def segment_with_details(self, messages: List[Dict[str, Any]]) -> TopicSegmentResult:
        """
        Segment messages and return the similarity matrix plus metadata.

        ``messages`` can be either OpenAI-style role/content messages or records
        containing user/assistant content fields.
        """

        if not messages:
            return TopicSegmentResult([], [], 0, False)

        records = self.messages_to_turn_records(messages)
        total_tokens = self.count_messages_tokens(messages)
        if total_tokens <= self.token_threshold or len(records) <= 1:
            return TopicSegmentResult(
                segments=[
                    TopicSegment(
                        topic_id=0,
                        messages=list(messages),
                        records=records,
                        start_record_index=0,
                        end_record_index=max(0, len(records) - 1),
                        token_count=total_tokens,
                    )
                ],
                similarity_matrix=[],
                total_tokens=total_tokens,
                segmented=False,
            )

        record_texts = [record["text"] for record in records]
        embeddings = self.embed_records(record_texts)
        similarity_matrix = self.compute_similarity_matrix(embeddings)
        boundaries = self.find_topic_boundaries(similarity_matrix)
        segments = self.build_segments(records, boundaries)

        return TopicSegmentResult(
            segments=segments,
            similarity_matrix=similarity_matrix,
            total_tokens=total_tokens,
            segmented=len(segments) > 1,
        )

    def messages_to_turn_records(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fold messages into user+assistant turn records.

        Records that already contain both sides are preserved as one turn. For
        role/content messages, a user message and following assistant message are
        combined into a single record.
        """

        if not messages:
            return []

        if any(self._has_turn_content(message) for message in messages):
            return [
                self._record_from_pair(
                    user_content=self._first_present(message, self.USER_KEYS),
                    assistant_content=self._first_present(message, self.ASSISTANT_KEYS),
                    messages=[message],
                    index=index,
                )
                for index, message in enumerate(messages)
                if self._has_turn_content(message)
            ]

        records: List[Dict[str, Any]] = []
        pending_user: Optional[Dict[str, Any]] = None

        for message in messages:
            role = str(message.get("role", "")).lower()
            if role == "user":
                if pending_user is not None:
                    records.append(
                        self._record_from_pair(
                            user_content=pending_user.get("content", ""),
                            assistant_content="",
                            messages=[pending_user],
                            index=len(records),
                        )
                    )
                pending_user = message
            elif role == "assistant":
                if pending_user is None:
                    records.append(
                        self._record_from_pair(
                            user_content="",
                            assistant_content=message.get("content", ""),
                            messages=[message],
                            index=len(records),
                        )
                    )
                else:
                    records.append(
                        self._record_from_pair(
                            user_content=pending_user.get("content", ""),
                            assistant_content=message.get("content", ""),
                            messages=[pending_user, message],
                            index=len(records),
                        )
                    )
                    pending_user = None
            else:
                if pending_user is not None:
                    records.append(
                        self._record_from_pair(
                            user_content=pending_user.get("content", ""),
                            assistant_content="",
                            messages=[pending_user],
                            index=len(records),
                        )
                    )
                    pending_user = None
                records.append(
                    self._record_from_pair(
                        user_content=str(message.get("content", "")),
                        assistant_content="",
                        messages=[message],
                        index=len(records),
                    )
                )

        if pending_user is not None:
            records.append(
                self._record_from_pair(
                    user_content=pending_user.get("content", ""),
                    assistant_content="",
                    messages=[pending_user],
                    index=len(records),
                )
            )

        return [record for record in records if record["text"].strip()]

    def count_messages_tokens(self, messages: List[Dict[str, Any]]) -> int:
        return sum(self.count_text_tokens(self.message_to_text(message)) for message in messages)

    def count_text_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.tokenizer is not None:
            return max(1, len(self.tokenizer.encode(text)))
        return max(1, len(text))

    def message_to_text(self, message: Dict[str, Any]) -> str:
        if self._has_turn_content(message):
            user_content = self._first_present(message, self.USER_KEYS)
            assistant_content = self._first_present(message, self.ASSISTANT_KEYS)
            return self.turn_to_text(user_content, assistant_content)

        role = message.get("role", "")
        content = message.get("content", "")
        return f"{role}: {content}" if role else str(content)

    def embed_records(self, texts: List[str]) -> List[List[float]]:
        if self.embedding_fn is not None:
            embeddings = self.embedding_fn(texts)
        elif self.embedding_client is not None:
            embeddings = self.embedding_client.get_embeddings(texts)
        else:
            raise ValueError("长文本主题划分需要配置 embedding_client 或 embedding_fn")

        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()

        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise ValueError(
                f"embedding 结果维度不符合预期: expected ({len(texts)}, dim)"
            )

        normalized_embeddings: List[List[float]] = []
        for embedding in embeddings:
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            if not isinstance(embedding, list):
                raise ValueError("embedding 结果必须是二维向量")
            normalized_embeddings.append([float(value) for value in embedding])

        return normalized_embeddings

    def compute_similarity_matrix(self, embeddings: List[List[float]]) -> List[List[float]]:
        matrix: List[List[float]] = []
        norms = [self._vector_norm(embedding) for embedding in embeddings]

        for row_index, row_embedding in enumerate(embeddings):
            row: List[float] = []
            for col_index, col_embedding in enumerate(embeddings):
                denominator = norms[row_index] * norms[col_index]
                if denominator <= 1e-12:
                    row.append(0.0)
                else:
                    row.append(self._dot(row_embedding, col_embedding) / denominator)
            matrix.append(row)

        return matrix

    def find_topic_boundaries(self, similarity_matrix: List[List[float]]) -> List[int]:
        """
        Return start indexes for each topic segment.

        Topic segmentation must preserve conversation order, so boundaries are
        detected from adjacent turn similarities in the full similarity matrix.
        """

        boundaries = [0]
        for index in range(len(similarity_matrix) - 1):
            adjacent_similarity = float(similarity_matrix[index][index + 1])
            if adjacent_similarity < self.similarity_threshold:
                boundaries.append(index + 1)
        return boundaries

    def build_segments(self, records: List[Dict[str, Any]], boundaries: List[int]) -> List[TopicSegment]:
        segments: List[TopicSegment] = []
        starts = boundaries or [0]

        for topic_id, start in enumerate(starts):
            end = starts[topic_id + 1] if topic_id + 1 < len(starts) else len(records)
            segment_records = records[start:end]
            segment_messages = [
                message
                for record in segment_records
                for message in record.get("messages", [])
            ]
            token_count = sum(self.count_text_tokens(record["text"]) for record in segment_records)
            segments.append(
                TopicSegment(
                    topic_id=topic_id,
                    messages=segment_messages,
                    records=segment_records,
                    start_record_index=start,
                    end_record_index=end - 1,
                    token_count=token_count,
                )
            )

        return segments

    def _record_from_pair(
        self,
        user_content: Any,
        assistant_content: Any,
        messages: List[Dict[str, Any]],
        index: int,
    ) -> Dict[str, Any]:
        user_content = "" if user_content is None else str(user_content)
        assistant_content = "" if assistant_content is None else str(assistant_content)
        return {
            "index": index,
            "user_content": user_content,
            "assistant_content": assistant_content,
            "text": self.turn_to_text(user_content, assistant_content),
            "messages": messages,
        }

    def turn_to_text(self, user_content: str, assistant_content: str) -> str:
        parts = []
        if user_content:
            parts.append(f"user: {user_content}")
        if assistant_content:
            parts.append(f"assistant: {assistant_content}")
        return "\n".join(parts)

    def _has_turn_content(self, message: Dict[str, Any]) -> bool:
        return any(key in message for key in self.USER_KEYS + self.ASSISTANT_KEYS)

    def _first_present(self, data: Dict[str, Any], keys: Sequence[str]) -> str:
        for key in keys:
            value = data.get(key)
            if value is not None:
                return str(value)
        return ""

    def _dot(self, left: List[float], right: List[float]) -> float:
        return sum(left_value * right_value for left_value, right_value in zip(left, right))

    def _vector_norm(self, vector: List[float]) -> float:
        return sum(value * value for value in vector) ** 0.5

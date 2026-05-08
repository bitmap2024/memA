#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Memory extract pipeline.

This module owns the low-level extract-store logic:
normalize chat records, compress text, chunk by token budget, extract memories
with LLM, embed them, and directly insert them into Qdrant.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from loguru import logger
try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover - optional runtime dependency
    AutoTokenizer = None

from api.emb_api import EmbeddingClientPool
from api.qdrant_api import QdrantMemoryClient
from config.config import Config
from data_loader.message_normalizer import MessageNormalizer
from extract_mem.memory_extractor import LLmExtractor
from extract_mem.text_compressor import TextCompressor


MEMORY_TYPE_MAPPING = {
    "事实记忆": "factual",
    "事实": "factual",
    "Factual": "factual",
    "偏好记忆": "preference",
    "偏好": "preference",
    "Preference": "preference",
    "能力与发展": "ability",
    "能力": "ability",
    "发展": "ability",
    "Ability": "ability",
    "Ability & Development": "ability",
    "社会关系": "relationship",
    "关系": "relationship",
    "Relationship": "relationship",
    "性格与画像": "portrait",
    "性格": "portrait",
    "画像": "portrait",
    "Portrait": "portrait",
}


class TokenChunker:
    """Chunk messages by a token budget while preserving chat order."""

    def __init__(self, max_tokens: int, tokenizer: Optional[Any] = None):
        self.max_tokens = max(1, int(max_tokens))
        self.tokenizer = tokenizer

    def count_text_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.tokenizer is not None:
            return max(1, len(self.tokenizer.encode(text)))
        return max(1, len(text))

    def count_message_tokens(self, message: Dict[str, Any]) -> int:
        role = message.get("role", "")
        content = message.get("content", "")
        return self.count_text_tokens(f"{role}: {content}")

    def chunk_messages(self, messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        chunks: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        current_tokens = 0

        for message in messages:
            token_count = self.count_message_tokens(message)
            if current and current_tokens + token_count > self.max_tokens:
                chunks.append(current)
                current = []
                current_tokens = 0

            current.append(message)
            current_tokens += token_count

        if current:
            chunks.append(current)

        return chunks


class MemoryExtractPipeline:
    """Low-level extract and direct-store pipeline."""

    memory_type_mapping = MEMORY_TYPE_MAPPING

    def __init__(
        self,
        config: Config = None,
        embedding_client: Optional[Any] = None,
        vector_store: Optional[QdrantMemoryClient] = None,
        extractor: Optional[LLmExtractor] = None,
    ):
        self.config = config or Config
        self.message_normalizer = MessageNormalizer()
        self.memory_extractor = extractor or LLmExtractor()

        if embedding_client is None:
            self.embedding_pool = EmbeddingClientPool(
                host=self.config.embedding.HOST,
                port=self.config.embedding.PORT,
                pool_size=5,
                timeout=100,
            )
            self.embedding_client = self.embedding_pool.get_client()
        else:
            self.embedding_pool = None
            self.embedding_client = embedding_client

        self.vector_store = vector_store or QdrantMemoryClient(
            collection_name=self.config.qdrant.COLLECTION,
            vector_size=self.config.qdrant.VECTOR_SIZE,
            distance=self.config.qdrant.DISTANCE,
        )

        compressor_path = self.config.pre_compressor.MODEL_PATH
        if compressor_path and os.path.exists(compressor_path):
            self.text_compressor = TextCompressor(model_path=compressor_path)
        else:
            self.text_compressor = None
            logger.warning("PreCompressor 模型路径未配置或不存在，抽取任务将跳过文本压缩")

        self.tokenizer = self._load_tokenizer(compressor_path)

    def _load_tokenizer(self, tokenizer_path: str):
        if not tokenizer_path or not os.path.exists(tokenizer_path):
            return None
        if AutoTokenizer is None:
            return None
        try:
            return AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception as e:
            logger.warning(f"Tokenizer 加载失败，将使用字符数估算 token: {e}")
            return None

    def normalize_messages(self, chat_records: List[Dict], session_start_time: str) -> List[Dict]:
        if not chat_records:
            return []
        try:
            messages_with_ts = []
            for record in chat_records:
                record_copy = record.copy()
                record_copy["time_stamp"] = session_start_time
                messages_with_ts.append(record_copy)
            return self.message_normalizer.normalize_messages(messages_with_ts)
        except Exception as e:
            logger.warning(f"Message Normalizer 规范化失败，使用原始记录: {e}")
            return chat_records

    def compress_messages(self, messages: List[Dict], rate: float = 0.5) -> List[Dict]:
        if self.text_compressor is None:
            return messages
        try:
            return self.text_compressor.compress_and_annotate(messages.copy(), rate=rate)
        except Exception as e:
            logger.warning(f"Message Compressor 压缩失败，使用原始文本: {e}")
            return messages

    def chunk_messages(self, messages: List[Dict], chunk_tokens: int) -> List[List[Dict]]:
        return TokenChunker(max_tokens=chunk_tokens, tokenizer=self.tokenizer).chunk_messages(messages)

    def extract_memories(self, messages: List[Dict]) -> List[Dict]:
        try:
            text = self.messages_to_text(messages)
            if not text.strip():
                return []
            response = self.memory_extractor.extract(text)
            return self.parse_memory_response(response)
        except Exception as e:
            logger.warning(f"记忆提取失败: {e}")
            return []

    def store_memories(
        self,
        memories: List[Dict],
        child_id: str,
        agent_id: str,
        session_id: str,
        session_start_time: str,
    ) -> int:
        stored_count = 0
        current_time = datetime.now(ZoneInfo("UTC")).isoformat()

        for memory in memories:
            try:
                memory_content = memory.get("content", "")
                memory_type = memory.get("type", "other")
                if not self._should_store_memory(memory_content):
                    continue

                memory_id = self.generate_memory_id(child_id, agent_id, memory_content)
                embedding = self.embedding_client.get_embeddings([memory_content])[0].tolist()
                payload = {
                    "memory_id": memory_id,
                    "child_id": child_id,
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "memory_content": memory_content,
                    "memory_type": self.memory_type_mapping.get(memory_type, memory_type),
                    "created_at": session_start_time or current_time,
                    "updated_at": current_time,
                    "merged": False,
                    "metion_count": 1,
                    "update_queue": [],
                }
                self.vector_store.upsert_one(
                    point_id=memory_id,
                    vector=embedding,
                    payload=payload,
                )
                stored_count += 1
            except Exception as e:
                logger.warning(f"存储记忆失败: {e}")

        return stored_count

    def process_history(
        self,
        history: Dict[str, Any],
        compress_rate: float = 0.5,
        chunk_tokens: int = 2048,
    ) -> Dict[str, int]:
        result = {
            "sessions": 0,
            "chunks": 0,
            "memories": 0,
            "stored": 0,
        }
        child_id = history.get("child_id")
        agent_id = history.get("agent_id")
        sessions = history.get("sessions", [])
        result["sessions"] = len(sessions)

        for session in sessions:
            session_id = session.get("session_id")
            session_start_time = session.get("session_start_time")
            chat_records = session.get("chat_records", [])
            if not chat_records:
                continue

            normalized = self.normalize_messages(chat_records, session_start_time)
            compressed = self.compress_messages(normalized, rate=compress_rate)
            chunks = self.chunk_messages(compressed, chunk_tokens=chunk_tokens)
            result["chunks"] += len(chunks)

            for chunk in chunks:
                memories = self.extract_memories(chunk)
                result["memories"] += len(memories)
                if memories:
                    result["stored"] += self.store_memories(
                        memories=memories,
                        child_id=child_id,
                        agent_id=agent_id,
                        session_id=session_id,
                        session_start_time=session_start_time,
                    )

        return result

    def messages_to_text(self, messages: List[Dict]) -> str:
        lines = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def parse_memory_response(self, response: str) -> List[Dict]:
        memories: List[Any] = []
        try:
            response = response.strip()
            try:
                data = json.loads(response)
                if isinstance(data, list):
                    memories = data
                elif isinstance(data, dict) and "memories" in data:
                    memories = data["memories"]
            except json.JSONDecodeError:
                json_obj_match = re.search(r"\{.*\}", response, re.DOTALL)
                if json_obj_match:
                    try:
                        data = json.loads(json_obj_match.group())
                        if isinstance(data, dict) and "memories" in data:
                            memories = data["memories"]
                    except json.JSONDecodeError:
                        pass

                if not memories:
                    json_arr_match = re.search(r"\[.*\]", response, re.DOTALL)
                    if json_arr_match:
                        data = json.loads(json_arr_match.group())
                        if isinstance(data, list):
                            memories = data
        except Exception as e:
            logger.warning(f"解析记忆响应失败: {e}")

        standardized = []
        for mem in memories:
            if isinstance(mem, str):
                standardized.append({"content": mem, "type": "other"})
            elif isinstance(mem, dict):
                standardized.append(
                    {
                        "content": mem.get("content") or mem.get("memory") or mem.get("text", ""),
                        "type": mem.get("memory_type") or mem.get("type") or mem.get("category", "other"),
                    }
                )

        return standardized

    def generate_memory_id(self, child_id: str, agent_id: str, content: str) -> str:
        unique_str = f"{child_id}_{agent_id}_{content}_{datetime.now().timestamp()}"
        return hashlib.md5(unique_str.encode()).hexdigest()

    def _should_store_memory(self, memory_content: str) -> bool:
        if not memory_content:
            return False
        excluded_names = ("卡卡", "罗西", "可豆", "贝贝")
        return not any(name in memory_content for name in excluded_names)

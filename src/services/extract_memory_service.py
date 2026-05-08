#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ExtractMemoryService - offline memory extraction service.

The service owns scheduling/orchestration. The extract-store implementation
lives in extract_mem.memory_extract_pipeline.
"""

from __future__ import annotations

import concurrent.futures
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from loguru import logger
from tqdm import tqdm

from config.config import Config
from data_loader.read_messages import ReadMessages
from extract_mem.memory_extract_pipeline import MEMORY_TYPE_MAPPING, MemoryExtractPipeline


class ExtractMemoryService:
    """Offline extraction entrypoint used by Argo jobs."""

    memory_type_mapping = MEMORY_TYPE_MAPPING

    def __init__(self, config: Config = None, pipeline: Optional[MemoryExtractPipeline] = None):
        self.config = config or Config
        self.read_messages = ReadMessages(config=self.config)
        self.extract_pipeline = pipeline or MemoryExtractPipeline(config=self.config)

    def offline_extract_memories(
        self,
        start_time: datetime = None,
        end_time: datetime = None,
        time_range_hours: int = 24,
        compress_rate: float = 0.5,
        chunk_tokens: int = 2048,
        max_workers: int = 4,
        batch_size: int = 10,
    ) -> Dict[str, Any]:
        logger.info("=" * 50)
        logger.info("开始离线记忆抽取任务")
        logger.info(
            f"参数: time_range_hours={time_range_hours}, compress_rate={compress_rate}, "
            f"chunk_tokens={chunk_tokens}, max_workers={max_workers}, batch_size={batch_size}"
        )
        logger.info("=" * 50)

        if end_time is None:
            end_time = datetime.now(ZoneInfo("Asia/Shanghai"))
        if start_time is None:
            start_time = end_time - timedelta(hours=time_range_hours)

        result = {
            "total_users": 0,
            "processed_users": 0,
            "failed_users": 0,
            "total_sessions": 0,
            "total_chunks": 0,
            "total_memories": 0,
            "stored_memories": 0,
            "errors": [],
        }
        result_lock = threading.Lock()

        try:
            user_ids = self._get_user_ids()
            result["total_users"] = len(user_ids)
            if not user_ids:
                logger.warning("没有找到用户，抽取任务结束")
                return result

            for batch_start in range(0, len(user_ids), batch_size):
                batch_users = user_ids[batch_start : batch_start + batch_size]
                batch_num = batch_start // batch_size + 1
                total_batches = (len(user_ids) + batch_size - 1) // batch_size
                logger.info(f"处理抽取批次 {batch_num}/{total_batches}，用户数: {len(batch_users)}")

                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_user = {
                        executor.submit(
                            self._process_single_user,
                            child_id=user["child_id"],
                            agent_id=user["agent_id"],
                            start_time=start_time,
                            end_time=end_time,
                            compress_rate=compress_rate,
                            chunk_tokens=chunk_tokens,
                        ): user
                        for user in batch_users
                    }

                    for future in tqdm(
                        concurrent.futures.as_completed(future_to_user),
                        total=len(batch_users),
                        desc=f"抽取批次 {batch_num}",
                    ):
                        user = future_to_user[future]
                        try:
                            user_result = future.result(timeout=300)
                        except concurrent.futures.TimeoutError:
                            user_result = {
                                "success": False,
                                "error": f"抽取用户 {user.get('child_id')} 超时",
                            }
                        except Exception as e:
                            user_result = {
                                "success": False,
                                "error": f"抽取用户 {user.get('child_id')} 失败: {e}",
                            }

                        with result_lock:
                            if user_result.get("success"):
                                result["processed_users"] += 1
                                result["total_sessions"] += user_result.get("sessions", 0)
                                result["total_chunks"] += user_result.get("chunks", 0)
                                result["total_memories"] += user_result.get("memories", 0)
                                result["stored_memories"] += user_result.get("stored", 0)
                            else:
                                result["failed_users"] += 1
                                if user_result.get("error"):
                                    result["errors"].append(user_result["error"])
        except Exception as e:
            logger.error(f"离线记忆抽取任务失败: {e}")
            result["errors"].append(str(e))

        logger.info(f"离线记忆抽取任务完成: {result}")
        return result

    def process_single_user_sync(
        self,
        child_id: str,
        agent_id: str,
        start_time: datetime = None,
        end_time: datetime = None,
        time_range_hours: int = 24,
        compress_rate: float = 0.5,
        chunk_tokens: int = 2048,
    ) -> Dict[str, Any]:
        if end_time is None:
            end_time = datetime.now(ZoneInfo("Asia/Shanghai"))
        if start_time is None:
            start_time = end_time - timedelta(hours=time_range_hours)
        return self._process_single_user(
            child_id=child_id,
            agent_id=agent_id,
            start_time=start_time,
            end_time=end_time,
            compress_rate=compress_rate,
            chunk_tokens=chunk_tokens,
        )

    def _process_single_user(
        self,
        child_id: str,
        agent_id: str,
        start_time: datetime,
        end_time: datetime,
        compress_rate: float = 0.5,
        chunk_tokens: int = 2048,
    ) -> Dict[str, Any]:
        result = {
            "success": False,
            "child_id": child_id,
            "agent_id": agent_id,
            "sessions": 0,
            "chunks": 0,
            "memories": 0,
            "stored": 0,
            "error": None,
        }
        try:
            history = self._fetch_user_history(child_id, agent_id, start_time, end_time)
            if not history or not history.get("sessions"):
                result["success"] = True
                return result

            stats = self.extract_pipeline.process_history(
                history=history,
                compress_rate=compress_rate,
                chunk_tokens=chunk_tokens,
            )
            result.update(stats)
            result["success"] = True
            logger.info(
                f"用户 {child_id} 抽取完成: sessions={result['sessions']}, "
                f"chunks={result['chunks']}, memories={result['memories']}, stored={result['stored']}"
            )
        except Exception as e:
            result["error"] = f"用户 {child_id} 抽取失败: {e}"
            logger.error(result["error"])
        return result

    def _get_user_ids(self, child_id: str = None, agent_id: str = None) -> List[Dict[str, str]]:
        user_ids = self.read_messages.get_user_ids()
        if child_id is not None:
            user_ids = [u for u in user_ids if u.get("child_id") == child_id]
        if agent_id is not None:
            user_ids = [u for u in user_ids if u.get("agent_id") == agent_id]

        seen = set()
        deduped = []
        for user in user_ids:
            key = (user.get("child_id"), user.get("agent_id"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(user)
        return deduped

    def _fetch_user_history(
        self,
        child_id: str,
        agent_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> Optional[Dict]:
        return self.read_messages.get_single_user_history(
            child_id=child_id,
            agent_id=agent_id,
            start_time=start_time,
            end_time=end_time,
        )

    # Backward-compatible alias for older callers.
    def offline_process_memories(self, *args, **kwargs) -> Dict[str, Any]:
        return self.offline_extract_memories(*args, **kwargs)

    def _step1_normalize_messages(self, chat_records: List[Dict], session_start_time: str) -> List[Dict]:
        return self.extract_pipeline.normalize_messages(chat_records, session_start_time)

    def _step2_compress_text(self, messages: List[Dict], rate: float = 0.5) -> List[Dict]:
        return self.extract_pipeline.compress_messages(messages, rate)

    def _step3_chunk_messages(self, messages: List[Dict], chunk_tokens: int = 2048) -> List[List[Dict]]:
        return self.extract_pipeline.chunk_messages(messages, chunk_tokens)

    def _step4_extract_memories(self, chunk: List[Dict]) -> List[Dict]:
        return self.extract_pipeline.extract_memories(chunk)

    def _step5_store_memories(
        self,
        memories: List[Dict],
        child_id: str,
        agent_id: str,
        session_id: str,
        session_start_time: str,
    ) -> int:
        return self.extract_pipeline.store_memories(
            memories=memories,
            child_id=child_id,
            agent_id=agent_id,
            session_id=session_id,
            session_start_time=session_start_time,
        )

    def _messages_to_text(self, messages: List[Dict]) -> str:
        return self.extract_pipeline.messages_to_text(messages)

    def _parse_memory_response(self, response: str) -> List[Dict]:
        return self.extract_pipeline.parse_memory_response(response)

    def _generate_memory_id(self, child_id: str, agent_id: str, content: str) -> str:
        return self.extract_pipeline.generate_memory_id(child_id, agent_id, content)

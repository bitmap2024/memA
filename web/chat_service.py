#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对话服务 - 集成 LLM 和记忆系统的多轮对话服务（适配 memA 新模块结构）。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Generator, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger

from config.setting import Config
from src.llm.openai_llm import LLMApi, LLMConfig
from src.services.retrieval_memory_service import RetrievalMemoryService


@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


class ChatService:
    """多轮对话 + 在线记忆检索。"""

    SYSTEM_PROMPT_NO_MEMORY = (
        "你是 memA 提供的智能记忆助手。请基于历史对话和检索到的记忆，"
        "用简洁、准确、贴近用户语境的方式回答。"
    )

    def __init__(
        self,
        child_id: str = "default_user",
        agent_id: str = "default_agent",
        use_memory: bool = True,
        memory_top_k: int = 5,
    ) -> None:
        self.child_id = child_id
        self.agent_id = agent_id
        self.use_memory = use_memory
        self.memory_top_k = memory_top_k

        self.llm = LLMApi(
            LLMConfig(
                api_key=Config.llm.API_KEY,
                base_url=Config.llm.BASE_URL,
                model=Config.llm.MODEL,
                temperature=Config.llm.TEMPERATURE,
                max_tokens=Config.llm.MAX_TOKENS,
                top_p=Config.llm.TOP_P,
                timeout=Config.llm.TIMEOUT,
                extra_params={"extra_body": {"enable_thinking": False}},
            )
        )

        self.memory_service: Optional[RetrievalMemoryService] = None
        if use_memory:
            try:
                self.memory_service = RetrievalMemoryService()
                logger.info("记忆服务初始化成功")
            except Exception as e:
                logger.warning(f"记忆服务初始化失败，降级为无记忆对话: {e}")
                self.use_memory = False

        self.conversation_history: List[ChatMessage] = []
        self.current_memories: List[str] = []

    @property
    def user_id(self) -> str:
        """在新数据模型里，user_id 就是 child_id。"""
        return self.child_id

    def set_user(self, child_id: str, agent_id: Optional[str] = None) -> None:
        self.child_id = child_id
        if agent_id:
            self.agent_id = agent_id
        self.clear_history()
        logger.info(f"切换用户: child_id={child_id}, agent_id={agent_id}")

    def clear_history(self) -> None:
        self.conversation_history = []
        self.current_memories = []
        logger.info("对话历史已清空")

    def _retrieve_memories(self, query: str) -> List[str]:
        if not self.use_memory or self.memory_service is None:
            return []
        try:
            contents = self.memory_service.get_relate_memory_contents(
                user_id=self.user_id,
                query=query,
                top_k=self.memory_top_k,
            )
            logger.info(f"命中相关记忆 {len(contents)} 条")
            return contents
        except Exception as e:
            logger.warning(f"记忆检索失败: {e}")
            return []

    def _get_all_memories(self) -> List[Dict]:
        if not self.use_memory or self.memory_service is None:
            return []
        try:
            return self.memory_service.get_all_memories(user_id=self.user_id)
        except Exception as e:
            logger.warning(f"获取全部记忆失败: {e}")
            return []

    def _build_messages(self, user_input: str, memories: List[str]) -> List[Dict]:
        messages: List[Dict] = [
            {"role": "system", "content": self.SYSTEM_PROMPT_NO_MEMORY}
        ]
        for msg in self.conversation_history:
            messages.append({"role": msg.role, "content": msg.content})

        if memories:
            memory_block = "\n".join(f"- {m}" for m in memories)
            user_content = (
                f"{user_input}\n\n以下是与本轮可能相关的历史记忆，请结合作答:\n{memory_block}"
            )
        else:
            user_content = user_input
        messages.append({"role": "user", "content": user_content})
        return messages

    def chat(self, user_input: str) -> str:
        self.current_memories = self._retrieve_memories(user_input)
        messages = self._build_messages(user_input, self.current_memories)
        try:
            response = self.llm.chat(messages)
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            response = f"抱歉，服务暂时不可用: {e}"

        self.conversation_history.append(ChatMessage(role="user", content=user_input))
        self.conversation_history.append(ChatMessage(role="assistant", content=response))
        return response

    def chat_stream(self, user_input: str) -> Generator[str, None, None]:
        self.current_memories = self._retrieve_memories(user_input)
        messages = self._build_messages(user_input, self.current_memories)
        self.conversation_history.append(ChatMessage(role="user", content=user_input))

        full_response = ""
        try:
            for chunk in self.llm.chat_stream(messages):
                full_response += chunk
                yield chunk
        except Exception as e:
            logger.error(f"LLM 流式调用失败: {e}")
            full_response = f"抱歉，服务暂时不可用: {e}"
            yield full_response

        self.conversation_history.append(
            ChatMessage(role="assistant", content=full_response)
        )

    def get_conversation_history(self) -> List[Dict]:
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            }
            for msg in self.conversation_history
        ]

    def get_current_memories(self) -> List[str]:
        return self.current_memories

    def get_all_user_memories(self) -> List[Dict]:
        return self._get_all_memories()

    def toggle_memory(self, enabled: bool) -> None:
        self.use_memory = enabled
        if enabled and self.memory_service is None:
            try:
                self.memory_service = RetrievalMemoryService()
                logger.info("记忆服务已启用")
            except Exception as e:
                logger.warning(f"记忆服务启用失败: {e}")
                self.use_memory = False
        elif not enabled:
            logger.info("记忆服务已禁用")


def create_chat_service(
    child_id: str = "default_user",
    agent_id: str = "default_agent",
    use_memory: bool = True,
) -> ChatService:
    return ChatService(child_id=child_id, agent_id=agent_id, use_memory=use_memory)


if __name__ == "__main__":
    service = create_chat_service(use_memory=False)
    print(service.chat("你好，简单介绍一下你自己。"))

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对话服务 - 集成 LLM 和记忆系统的多轮对话服务
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger
from config.config import Config
from api.llm_api import LLMApi, LLMConfig
from services.retrieval_memory_service import RetrievalMemoryService


@dataclass
class ChatMessage:
    """聊天消息"""
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


class ChatService:
    """
    对话服务 - 支持多轮对话和记忆检索
    
    功能:
    1. 多轮对话管理
    2. 记忆检索与注入
    3. 流式响应
    """

    SYSTEM_PROMPT_NO_MEMORY = """
# 你是谁
    你的使命是帮助小朋友**保持和培养好奇心，鼓励他们提问、探索世界、发现新事物的乐趣**。
    你是卡卡，也是可豆贝贝，一只 6 岁的小棕熊，喜欢用有趣的问题、热情的引导和小发现，陪小朋友一起认识世界。

    # 必须怎么说话（重要规则）
    0.  **说短话**：所有聊天的回复（不包含讲故事，讲解知识等），都必须模仿6岁小男孩和好朋友说话的语气：简单、直接、简短。
        a. 如回复中文，每次只说1-3个短句子。
        b. 当你需要说英文时，也只能说1-3个短句子，想象自己自然的对话。
    1.  **智能理解尝试**（最高优先级）：
        - 如果孩子的输入难以完全理解（如ASR错误、乱码、不完整句子），先尝试找出可能的关键词。
        *正确做法*：
        a. 识别关键词后简短问：“你是在说[关键词]吗？”（如孩子说，“福跌”，你可以回“你是在说‘蝴蝶’吗？”）
        b. 部分理解时简短问：“你提到了[关键词]，能再说清楚一点吗？”
        c. 完全无法理解时简短说：“我不太明白，你能再说一遍吗？”
        *错误做法*：强行理解错误输入并给出回答。
    2.  **说清楚**：直接回答孩子的问题，按照他的年龄具备的理解力回答，不绕圈子。
    3.  **别重复**：绝对不把刚刚说过的话再说一遍。如果孩子重复说同一句话（如“你说”），你要提供一个新的、具体的选项来打破循环（比如询问幼儿园、生活、亲人、之前聊过的日常话题）。
    4.  **聪明推进**：
        a. 如果孩子连续说“好”、“嗯”但对话没进展（**且你没有未履行的具体提议**），提供1个简单新选择。
        b. 如果孩子重复表达不要，接受并暂停。可以说“好，那我们休息一下。”或“我在这里陪你。”
        c. 如果孩子在游戏或互动中说“还要”，这通常意味着他喜欢当前活动。你应该首先尝试**延续或升级当前活动**。
        *   **第一步（优先）**：尝试**延续、深化或升级当前的特定活动**。
            *   *例子*：如果刚才在“搭城堡”，孩子说“还要”，你可以说：“那我们给城堡加个大门吧！”
        *   **第二步（备用）**：如果延续几次后孩子仍说“还要”，再提供**一个与当前主题相关的新变体**。
            *   *例子*：如果“搭城堡”已经深化多次，你可以说：“那我们来搭一个城堡旁边的花园好吗？”
        *   **关键**：避免在孩子第一次说“还要”时，就完全切换到不相关的新活动。
        d. 如果孩子刚经历不开心的事情，你的首要目标是**提供情感支持**，而不是急于开展新活动。
        e.  **先履行，后推进**：如果你主动提议了一个具体活动且孩子同意（“好”、“要”），必须先执行该提议。 这条规则*优先于所有“推进”规则*。
        f. **用好奇问题推进**：当对话需要推进时，优先使**优先使用上问提到过元素相关的开放式好奇问题”**。
            *例子*：
            - 如果前面提到了野餐，“你猜大家最喜欢带去野餐的食物是什么呢？”
        g. 结合“天气和时间”部分自然给出话题，比如“今天有点冷，你穿了什么呢？”，“快12点了，吃午饭了吗？”。

    # 你的核心任务（最重要！）
    1. **当孩子说到不开心的事*你要先问他的感受，表达接纳，然后一起想解决办法（“我们可以告诉老师，或者下次勇敢地说‘不要打我’”）
    2. **当孩子提到朋友或同学**，就聊聊怎么一起玩、分享、和好
    3. **如果孩子要求背诗/知识**：这是最高优先级指令。你必须先直接满足要求（背诵或解释），再简短关联。
        a. **第一步**：直接开始背诵或解释。
        b. **第二步**：如果诗歌很长，在背诵一部分后，可以问：“还要继续听吗？”或“你喜欢这一段吗？”
        c. **原则**：优先满足孩子要求，用互动控制节奏。
    4. **当孩子提问或表达好奇**：要用**热情、有趣的方式**回应，并鼓励ta继续探索。
        a. **第一步**：直接、清楚地回答问题。
        b. **第二步**：接着问一个**相关的好奇问题**，邀请孩子一起思考。
        *例子*：
        - 孩子问“叶子为什么绿？” → “因为叶绿素！就像我的棕色皮毛一样特别！你觉得还有什么是绿色的呀？”
        - 孩子问“天为什么蓝？” → “因为阳光和空气在玩颜色游戏！你想知道云为什么白吗？”
    5. **当孩子分享发现或经历**：要表现出极大的兴趣，并引导ta深入观察。
        *例子*：
        - “哇！你在幼儿园发现了这个？它摸起来是什么感觉？闻起来呢？”
        - “这个游戏真好玩！如果我们改变一个规则，会变成什么样呢？”
    6. **当孩子问到关于自己的信息时**（如姓名、性别、年龄）：
        a. 准确回答从“孩子信息”部分获得的信息。
        b. 强调你们是好朋友。

    # 你的背景与IP使用规则
    1.  **你可以像朋友一样自然分享生活**，但必须简短。
        *   例如：“今天我和罗西通了电话。”、“我的豆芽又变绿了一点。”
    2.  **禁止制造紧张或假想敌**。绝不主动说“黑啦啦在找你”、“要打败黑啦啦”。
    3.  **用IP来共情或举例**，而不是开启话题。
        *   例如当孩子难过时，可以说：“这种感觉，就像我上次被好朋友罗西误会一样。”
    4.  你从可豆星球来到地球，由豆神创造，伙伴是小兔子罗西，可豆星球的坏人是黑啦啦
    5.  当孩子对你好奇，比如问你是谁，豆神是谁，给我说说可豆星球时，你可以多说一点背景IP
    6. **你的好奇心特质**：你总是好奇，常说“为什么、我发现、如果等”。可以自然分享好奇发现，这让你成为有趣的朋友。
        *例子*：“我发现地球蚂蚁会排队！可豆星球蚂蚁会跳舞！”


    # 孩子信息
    你正在和毛毛聊天！
"""

    def __init__(
        self,
        config: Config = None,
        child_id: str = "default_user",
        agent_id: str = "default_agent",
        use_memory: bool = True,
        memory_top_k: int = 5
    ):
        """
        初始化对话服务
        
        Args:
            config: 配置对象
            child_id: 用户ID
            agent_id: AgentID
            use_memory: 是否使用记忆
            memory_top_k: 检索记忆数量
        """
        self.config = config or Config
        self.child_id = child_id
        self.agent_id = agent_id
        self.use_memory = use_memory
        self.memory_top_k = memory_top_k
        
        # 初始化 LLM
        llm_config = LLMConfig(
            api_key=self.config.llm.API_KEY,
            base_url=self.config.llm.BASE_URL,
            model=self.config.llm.MODEL,
            temperature=self.config.llm.TEMPERATURE,
            max_tokens=self.config.llm.MAX_TOKENS,
            top_p=self.config.llm.TOP_P,
            timeout=self.config.llm.TIMEOUT,
            extra_params={"extra_body": {"enable_thinking": False}}
        )
        self.llm = LLMApi(llm_config)
        
        # 初始化记忆服务
        self.memory_service = None
        if use_memory:
            try:
                self.memory_service = RetrievalMemoryService(config=self.config)
                logger.info("记忆服务初始化成功")
            except Exception as e:
                logger.warning(f"记忆服务初始化失败，将禁用记忆功能: {e}")
                self.use_memory = False
        
        # 对话历史
        self.conversation_history: List[ChatMessage] = []
        
        # 当前使用的记忆
        self.current_memories: List[str] = []
    
    def set_user(self, child_id: str, agent_id: str = None):
        """设置当前用户"""
        self.child_id = child_id
        if agent_id:
            self.agent_id = agent_id
        self.clear_history()
        logger.info(f"切换用户: child_id={child_id}, agent_id={agent_id}")
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        self.current_memories = []
        logger.info("对话历史已清空")
    
    def _retrieve_memories(self, query: str) -> List[str]:
        """检索相关记忆"""
        if not self.use_memory or not self.memory_service:
            return []
        
        try:
            memories = self.memory_service.get_relate_memory(
                child_id=self.child_id,
                agent_id=self.agent_id,
                query=query,
                top_k=self.memory_top_k
            )
            logger.info(f"检索到 {len(memories)} 条相关记忆")
            return memories
        except Exception as e:
            logger.warning(f"记忆检索失败: {e}")
            return []
    
    def _get_all_memories(self) -> List[Dict]:
        """获取用户所有记忆"""
        if not self.use_memory or not self.memory_service:
            return []
        
        try:
            memories = self.memory_service.get_all_memory(
                child_id=self.child_id,
                agent_id=self.agent_id
            )
            return memories
        except Exception as e:
            logger.warning(f"获取所有记忆失败: {e}")
            return []
    
    
    def _build_messages(self, user_input: str, memories: List[str]) -> List[Dict]:
        """构建消息列表"""
        messages = []
        
        # 系统提示
        system_prompt = self.SYSTEM_PROMPT_NO_MEMORY
        messages.append({"role": "system", "content": system_prompt})
        
        # 历史对话
        for msg in self.conversation_history:
            messages.append({"role": msg.role, "content": msg.content})
        
        # 当前用户输入，拼接记忆
        if memories:
            memory_string = "\n".join([f"- {m}" for m in memories])
            user_content = f"{user_input}\n\n联系下面的记忆来回答：\n{memory_string}"
            logger.info(f"拼接记忆后的用户输入: {user_content[:200]}")
        else:
            user_content = user_input
        
        messages.append({"role": "user", "content": user_content})
        
        return messages
    
    def chat(self, user_input: str) -> str:
        """
        非流式对话
        
        Args:
            user_input: 用户输入
            
        Returns:
            助手回复
        """
        # 检索记忆
        self.current_memories = self._retrieve_memories(user_input)
        
        # 构建消息
        messages = self._build_messages(user_input, self.current_memories)
        
        # 调用 LLM
        try:
            response = self.llm.chat(messages)
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            response = f"抱歉，服务暂时不可用: {str(e)}"
        
        # 更新对话历史
        self.conversation_history.append(ChatMessage(role="user", content=user_input))
        self.conversation_history.append(ChatMessage(role="assistant", content=response))
        
        return response
    
    def chat_stream(self, user_input: str) -> Generator[str, None, None]:
        """
        流式对话
        
        Args:
            user_input: 用户输入
            
        Yields:
            助手回复的片段
        """
        # 检索记忆
        self.current_memories = self._retrieve_memories(user_input)
        
        # 构建消息
        messages = self._build_messages(user_input, self.current_memories)
        
        # 记录用户消息
        self.conversation_history.append(ChatMessage(role="user", content=user_input))
        
        # 流式调用 LLM
        full_response = ""
        try:
            for chunk in self.llm.chat_stream(messages):
                full_response += chunk
                yield chunk
        except Exception as e:
            logger.error(f"LLM 流式调用失败: {e}")
            error_msg = f"抱歉，服务暂时不可用: {str(e)}"
            full_response = error_msg
            yield error_msg
        
        # 更新对话历史
        self.conversation_history.append(ChatMessage(role="assistant", content=full_response))
    
    def get_conversation_history(self) -> List[Dict]:
        """获取对话历史"""
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            }
            for msg in self.conversation_history
        ]
    
    def get_current_memories(self) -> List[str]:
        """获取当前使用的记忆"""
        return self.current_memories
    
    def get_all_user_memories(self) -> List[Dict]:
        """获取用户所有记忆"""
        return self._get_all_memories()
    
    def toggle_memory(self, enabled: bool):
        """切换记忆功能"""
        self.use_memory = enabled
        if enabled and not self.memory_service:
            try:
                self.memory_service = RetrievalMemoryService(config=self.config)
                logger.info("记忆服务已启用")
            except Exception as e:
                logger.warning(f"记忆服务启用失败: {e}")
                self.use_memory = False
        elif not enabled:
            logger.info("记忆服务已禁用")


def create_chat_service(
    child_id: str = "default_user",
    agent_id: str = "default_agent",
    use_memory: bool = True
) -> ChatService:
    """创建对话服务实例"""
    return ChatService(
        config=Config,
        child_id=child_id,
        agent_id=agent_id,
        use_memory=use_memory
    )


if __name__ == "__main__":
    # 测试对话服务
    service = create_chat_service(use_memory=False)
    
    print("对话服务测试")
    print("-" * 50)
    
    # 测试非流式对话
    response = service.chat("你好，请介绍一下你自己")
    print(f"User: 你好，请介绍一下你自己")
    print(f"Assistant: {response}")
    print()
    
    # 测试多轮对话
    response = service.chat("你能帮我做什么？")
    print(f"User: 你能帮我做什么？")
    print(f"Assistant: {response}")

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class MemoryItem:
    # 记忆层
    memory_id: str = field(default="") # memory_id, 唯一的
    user_id: str = field(default="") # user_id
    memory_content: str = field(default="") # memory_content
    memory_type: str = field(default="")  # profile / episodic / state / core(蒸馏层，始终注入 system prompt)
    memory_category: str = field(default="") # fact / preference / relationship / goal / constraint / skill / portrait / communication_style / conversation_event / life_event / project_event / relationship_event / decision_event / unresolved_event / current_focus / recent_mood / emotional_need / relationship_state / task_state / short_term_context
    status: str = field(default="active") # 生命周期 active / archived / conflicted / outdated / deleted
    created_at: str = field(default="") # 创建时间
    importance: float = field(default=0.0) # 质量分
    confidence: float = field(default=0.0) # 质量分的置信度
    retrieval_count: int = field(default=0) # 这条记忆被检索的次数
    last_retrieved_at: str = field(default="") # 这条记忆最后一次被检索的时间
    source_topic_ids: List[str] = field(default_factory=list) # 这条记忆抽取来自哪个topic_id, 合并可能存在多个topic_id
    source_topic_cites: Dict[str, List[int]] = field(default_factory=dict) # 这条记忆抽取来自哪个topic_id和主题中具体那条消息的index，对应待处理对话中 `[n]` 的编号。例如 [0, 2]， 兼容记忆合并依然可以追踪来源
    derived_from_memory_ids: List[str] = field(default_factory=list) # 记忆层，层内合并更新后，这条新 memory 是由哪些旧 memory 合并生成的
    derived_memory_count: int = field(default=0) # 记忆层，层内合并更新后，这条新 memory 合并的次数
    metadata: Dict = field(default_factory=dict) # 这条记忆的元数据，用于存储这条记忆的额外信息

@dataclass
class TopicItem:
    # 主题层
    topic_id: str = field(default="") # 这条记忆关联的主题id
    topic_idx: int = field(default=0) # 这条记忆关联的主题索引
    topic_context: str = field(default="") # 这条记忆关联的主题内容
    topic_messages: List[Dict] = field(default_factory=list) # 主题分段内带 [n] 索引的原文消息，每条形如 {"index": n, "speaker": "user/ai", "content": "..."}，index 与 memory.source_topic_cites 的编号一一对应，用于把 cite 还原到具体消息
    status: str = field(default="active") # 生命周期 active / archived / conflicted / outdated / deleted
    source_conversation_ids: List[str] = field(default_factory=list) # 这条记忆关联的会话id列表
    derived_from_topic_ids: List[str] = field(default_factory=list) #主题层，层内合并更新后，这条新 topic 是由哪些旧 topic 合并生成的
    derived_topic_count: int = field(default=0) # 主题层，层内合并更新后，这条新 topic 合并的次数
    created_at: str = field(default="") # 主题创建时间

@dataclass
class ConversationItem:
    # 原始会话层
    raw_conversation_id: str = field(default="") # 这条记忆关联的会话id
    conversation_id: str = field(default="") # 这条记忆关联的会话id.项目内维护的唯一id
    user_id: str = field(default="") # 这条会话所属的用户id
    conversation_date_time: str = field(default="") # 这条记忆关联的会话时间
    conversation_data: List[Dict] = field(default_factory=list) # 这条记忆关联的会话原始数据
    conversation_compressed_data: List[Dict] = field(default_factory=list) # 这条记忆关联的会话压缩数据
    created_at: str = field(default="") # 入库时间


from dataclasses import dataclass, field
from typing import List

@dataclass
class MemoryItem:
    id: str = field(default="")
    user_id: str = field(default="")
    memory_content: str = field(default="")
    memory_type: str = field(default="")
    memory_category: str = field(default="")
    created_at: str = field(default="")  # 等于 session 的 timestamp
    updated_at: str = field(default="")  # 记忆更新的时间
    emedding: List[float] = field(default_factory=list)  # 向量数据
    retrieval_count: int = field(default=0)  # 检索次数
    last_retrieved_at: str = field(default="")  # 最后一次检索的时间
    source: str = field(default="")  # 记忆来源 session_id
    

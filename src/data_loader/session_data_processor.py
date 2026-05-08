from typing import List, Dict, Optional
from datetime import datetime
from tqdm import tqdm
from loguru import logger
from utils.utils import convert_timestamp

class SessionDataProcessor:
    """
    会话数据处理器，用于将历史会话数据转换为记忆系统所需的格式。
    """
    
    def __init__(self, show_progress: bool = True):
        """
        初始化会话数据处理器。
        
        Args:
            show_progress: 是否显示进度条
        """
        self.show_progress = show_progress
    
    def _calculate_total_turns(self, sessions: List[List[Dict]]) -> int:
        """
        计算所有会话的总轮次数。
        
        Args:
            sessions: 会话列表
            
        Returns:
            总轮次数
        """
        total_turns = 0
        for session in sessions:
            # 确保第一条消息来自用户
            session_copy = session.copy()
            while session_copy and session_copy[0]["role"] != "user":
                session_copy.pop(0)
            num_turns = len(session_copy) // 2
            total_turns += num_turns
        return total_turns
    
    def _process_single_session(
        self, 
        session: List[Dict], 
        session_id: str, 
        date: str,
        session_idx: int,
        total_sessions: int
    ) -> List[List[Dict]]:
        """
        处理单个会话，将其转换为轮次消息列表。
        
        Args:
            session: 单个会话的消息列表
            session_id: 会话 ID
            date: 会话时间戳（已转换格式）
            session_idx: 当前会话索引
            total_sessions: 总会话数
            
        Returns:
            该会话的轮次消息列表
        """
        turns_messages = []
        
        # 确保第一条消息来自用户
        session = session.copy()
        while session and session[0]["role"] != "user":
            session.pop(0)
        
        num_turns = len(session) // 2
        
        for turn_idx in range(num_turns):
            # 提取一轮对话（用户 + 助手消息）
            turn_messages = session[turn_idx * 2 : turn_idx * 2 + 2]
            
            # 验证轮次结构
            if len(turn_messages) < 2:
                continue
            if turn_messages[0]["role"] != "user" or turn_messages[1]["role"] != "assistant":
                continue
            
            # 为每条消息添加时间戳和说话者信息
            processed_turn = []
            for msg in turn_messages:
                msg_copy = msg.copy()
                msg_copy["time_stamp"] = date
                
                # 如果没有说话者信息，添加默认值
                if "speaker_name" not in msg_copy:
                    msg_copy["speaker_name"] = "User" if msg_copy["role"] == "user" else "Assistant"
                if "speaker_id" not in msg_copy:
                    msg_copy["speaker_id"] = "speaker_a" if msg_copy["role"] == "user" else "speaker_b"
                
                processed_turn.append(msg_copy)
            
            # 标记是否为最后一轮（用于强制分段和提取）
            is_last_turn = (session_idx == total_sessions - 1 and turn_idx == num_turns - 1)
            
            turns_messages.append({
                "messages": processed_turn,
                "session_id": session_id,
                "is_last_turn": is_last_turn
            })
        
        return turns_messages
    
    def process_sessions(
        self, 
        sessions: List[List[Dict]], 
        session_ids: List[str],
        dates: List[str]
    ) -> List[Dict]:
        """
        处理多个会话，将其转换为记忆系统所需的格式。
        
        Args:
            sessions: 会话列表，每个会话包含多个对话轮次
            session_ids: 会话 ID 列表
            dates: 会话时间戳列表（将被转换为标准格式）
            
        Returns:
            处理后的轮次消息列表，每个元素包含:
                - messages: 消息列表（用户 + 助手）
                - session_id: 会话 ID
                - is_last_turn: 是否为最后一轮
        """
        logger.info("正在将时间戳转换为标准格式...")
        
        # 转换所有时间戳为标准格式
        converted_dates = [convert_timestamp(date) for date in dates]
        
        # 计算总轮次数用于进度条
        total_turns = self._calculate_total_turns(sessions)
        
        all_turns_messages = []
        
        # 创建进度条
        progress_bar = None
        if self.show_progress:
            progress_bar = tqdm(total=total_turns, desc="处理会话轮次")
        
        try:
            for session_idx, (session, session_id, date) in enumerate(
                zip(sessions, session_ids, converted_dates)
            ):
                session_turns = self._process_single_session(
                    session=session,
                    session_id=session_id,
                    date=date,
                    session_idx=session_idx,
                    total_sessions=len(sessions)
                )
                
                all_turns_messages.extend(session_turns)
                
                if progress_bar:
                    progress_bar.update(len(session_turns))
        finally:
            if progress_bar:
                progress_bar.close()
        
        logger.info(f"处理完成，共 {len(all_turns_messages)} 个轮次")
        return all_turns_messages
    
    def get_messages_only(
        self, 
        sessions: List[List[Dict]], 
        session_ids: List[str],
        dates: List[str]
    ) -> List[List[Dict]]:
        """
        处理会话并只返回消息列表（不包含元数据）。
        
        Args:
            sessions: 会话列表
            session_ids: 会话 ID 列表
            dates: 会话时间戳列表
            
        Returns:
            轮次消息列表，每个元素是一个包含用户和助手消息的列表
        """
        processed = self.process_sessions(sessions, session_ids, dates)
        return [item["messages"] for item in processed]


if __name__ == "__main__":
    sessions = [
        [
            {"role": "user", "content": "Hello, how are you?", "time_stamp": "2026-02-13 13:21:00"},
            {"role": "assistant", "content": "I'm fine, thank you!", "time_stamp": "2026-02-13 13:21:00"},
        ]
    ]
    session_ids = ["1234567890"]
    dates = ["2026-02-13 13:21:00"]
    processor = SessionDataProcessor()
    messages = processor.get_messages_only(sessions, session_ids, dates)
    print(messages)
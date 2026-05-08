from loguru import logger
from config.config import Config
from api.whale import WhaleAPI
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
from tqdm import tqdm

'''
从mysql的接口（whale提供）获取所有用户的历史聊天记录

返回格式：
{
    "child_id": "xxx",
    "agent_id": "xxx",
    "sessions": [
        {
            "session_id": "xxx",
            "session_start_time": "xxx",
            "chat_records": [
                {
                    "role": "user",
                    "content": "xxx",
                },
                {
                    "role": "assistant",
                    "content": "xxx",
                },
                ...
            ]
        }
    ]
}

session_id: 会话ID
session_start_time: 会话开始时间
chat_records: 聊天记录
role: 角色
content: 内容
time_stamp: 时间戳
'''

class ReadMessages:
    def __init__(self, config: Config):
        self.whale = WhaleAPI(config.whale.HOST)
    
    def get_user_ids(self)->List[Dict]:
        all_agent_info = self.whale.get_base_info()
        if len(all_agent_info) == 0:
            raise Exception("get_user_ids's data is []")
        user_ids = []
        seen_keys = set()
        for agent in all_agent_info:
            try:
                child_id = agent["child_info"]["child_id"]
                agent_id = agent["agent_id"]
                key = (child_id, agent_id)
                if key not in seen_keys:
                    seen_keys.add(key)
                    user_ids.append({"child_id": child_id, "agent_id": agent_id})
            except:
                logger.error(f"{agent} response format is error")
                continue
        logger.info(f"get_user_ids number: {len(user_ids)} users")
        return user_ids

    def get_single_user_history(self, child_id: str, agent_id: str, start_time: datetime, end_time: datetime)->Optional[Dict]:
        """
        获取单个用户的聊天记录
        
        Args:
            child_id: 用户 ID
            agent_id: Agent ID  
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            用户聊天记录，格式:
            {
                "child_id": "xxx",
                "agent_id": "xxx",
                "sessions": [...]
            }
            如果没有数据则返回 None
        """
        try:
            # 检查 agent_id 是否为空
            if not agent_id:
                logger.warning(f"Skipping query for child_id: {child_id} due to empty agent_id")
                return None
            
            chat_data_items = self.whale.query_chat_info(
                child_id=child_id,
                agent_id=agent_id,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                order="asc",
            )
            
            if not chat_data_items:
                return None
            
            if not isinstance(chat_data_items, List):
                raise Exception("query_chat_info's data is not list")
            
            sessions = defaultdict(
                lambda: {"session_start_time": None, "chat_records_item_list": []}
            )

            for item in chat_data_items:
                session_id = item["session_id"]
                if sessions[session_id]["session_start_time"] is None:
                    sessions[session_id]["session_start_time"] = item["request_time"]
                if (item.get("intent") in ["chat", "story", "music", "holiday", "weather", "time", "control", "profile"]):  # 只从chat聊天记录中抽取记忆
                    user = {"role": "user", "content": item["chat_records_item"].get("RequestContent", "")}
                    ai = {"role": "assistant", "content": item["chat_records_item"].get("ResponseContent", "")}
                    sessions[session_id]["chat_records_item_list"].append(user)
                    sessions[session_id]["chat_records_item_list"].append(ai)

            sessions_data = [
                {
                    "session_id": session_id,
                    "session_start_time": session_info["session_start_time"],
                    "chat_records": session_info["chat_records_item_list"],
                }
                for session_id, session_info in sessions.items()
                if session_info["chat_records_item_list"]  # 只保留有聊天记录的会话
            ]

            if not sessions_data:
                return None

            return {
                "child_id": child_id,
                "agent_id": agent_id,
                "sessions": sessions_data,
            }
            
        except Exception as e:
            logger.error(f"Error getting history for child_id: {child_id}, agent_id: {agent_id}. Error: {str(e)}")
            return None
    
    def get_user_history(self, start_time, end_time):
        """
        获取所有用户的聊天记录（批量接口）
        """
        histories = []
        pbar = tqdm(self.ids, desc="Processing user history")
        for id_item in pbar:
            child_id = id_item["child_id"]
            agent_id = id_item["agent_id"]
            
            history = self.get_single_user_history(child_id, agent_id, start_time, end_time)
            if history:
                histories.append(history)
            pbar.update(1)
        pbar.close()
        return histories
    
if __name__ == "__main__":
    config = Config()
    read_qa_api = ReadMessages(config)
    user_ids = read_qa_api.get_user_ids()
    print(user_ids[:3])
    for user_id in user_ids:
        history = read_qa_api.get_single_user_history(user_id["child_id"], user_id["agent_id"], datetime.now() - timedelta(days=1), datetime.now())
        if history:
            print(history)
            break
        
        
        
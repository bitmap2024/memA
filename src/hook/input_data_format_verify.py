from typing import Dict, Any, List

from loguru import logger

history_schema = {
    "user_id": str,
    "raw_conversation_id": str,
    "conversation_date_time": str,
    "conversation": List[Dict[str, Any]],
}


def verify_input_data_format(input: Dict[str, Any]) -> bool:
    """校验输入数据的数据格式。"""
    if not isinstance(input, dict):
        return False, [f"输入数据必须是 dict，实际为 {type(input).__name__}"]
    if not isinstance(input["user_id"], str):
        return False, [f"user_id 必须是 str，实际为 {type(input["user_id"]).__name__}"]
    if not isinstance(input["raw_conversation_id"], str):
        return False, [f"raw_conversation_id 必须是 str，实际为 {type(input["raw_conversation_id"]).__name__}"]
    if not isinstance(input["conversation_date_time"], str):
        return False, [f"conversation_date_time 必须是 str，实际为 {type(input["conversation_date_time"]).__name__}"]
    if not isinstance(input["conversation"], List):
        return False, [f"conversation 必须是 List，实际为 {type(input["conversation"]).__name__}"]
    if not isinstance(input["conversation"][0], Dict):
        return False, [f"conversation 的第一个元素必须是 dict，实际为 {type(input["conversation"][0]).__name__}"]
    return True, []


import re 
import json
import hashlib
import pytz
from datetime import datetime
import numpy as np

def convert_timestamp(timestamp: str) -> str:
    """
    将多种时间戳格式统一转换为 '%Y-%m-%d %H:%M:%S' 格式。
    
    支持的输入格式:
        - '2025/12/02 (Tue) 17:06' (带星期)
        - '2025/12/02 17:06'
        - '2025-12-02 17:06:00' (已是目标格式)
        - '2025-12-02 17:06'
        - '2025-12-02T17:06:00' (ISO 格式)
    
    Args:
        timestamp: 原始时间戳字符串
        
    Returns:
        转换后的时间戳字符串，格式为 '%Y-%m-%d %H:%M:%S'
    """
    timestamp = timestamp.strip()
    
    # 如果已经是目标格式，直接返回
    try:
        dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
        return timestamp
    except ValueError:
        pass
    
    # 处理带括号的星期格式: '2025/12/02 (Tue) 17:06'
    if '(' in timestamp and ')' in timestamp:
        timestamp_clean = timestamp.split('(')[0].strip() + ' ' + timestamp.split(')')[1].strip()
        dt = datetime.strptime(timestamp_clean, '%Y/%m/%d %H:%M')
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # 尝试其他常见格式
    formats = [
        '%Y/%m/%d %H:%M',
        '%Y-%m-%d %H:%M',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M:%S.%f',
        '%Y/%m/%d %H:%M:%S',
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(timestamp, fmt)
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            continue
    
    # 如果都无法解析，抛出异常
    raise ValueError(f"无法解析时间戳格式: {timestamp}")

def json_format(text):
    def __extract_json(text):
        match = re.search(r'{.*}', text, re.DOTALL)
        if match:
            return match.group(0)
        return None

    def __fix_json(text):
        text = text.replace('：', ':').replace('，', ',')
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r'"\{(.*?)\}"', r'"\1"', text)
        return text
    try:
        json_data = __extract_json(text)
        if json_data:
            fixed_json_data = __fix_json(json_data)
            python_object = json.loads(fixed_json_data)
            return python_object
        else:
            return {}
    except json.JSONDecodeError as e:
        return {}
    
def generate_unique_id(content):
    # 使用 SHA-256 哈希算法
    unique_id = hashlib.md5(content.encode()).hexdigest()
    return unique_id

def utc_convert_beijing(utc_time_str):
    """
    将 UTC 时间字符串转换为北京时间字符串
    
    Args:
        utc_time_str: UTC 时间字符串，ISO 格式
        
    Returns:
        北京时间字符串，格式为 '%Y-%m-%d %H:%M:%S'
    """
    # 解析为 datetime 对象
    utc_time = datetime.fromisoformat(utc_time_str)
    # 定义北京时间时区
    beijing_tz = pytz.timezone("Asia/Shanghai")
    # 转换为北京时间
    beijing_time = utc_time.astimezone(beijing_tz)
    beijing_time_str = beijing_time.strftime("%Y-%m-%d %H:%M:%S")
    return beijing_time_str

def format_beijing_display(time_str):
    """
    格式化北京时间用于显示
    
    Args:
        time_str: 时间字符串，可以是 ISO 格式或已格式化的字符串
        
    Returns:
        格式化后的北京时间字符串，格式为 '%Y-%m-%d %H:%M:%S'
    """
    if not time_str:
        return ""
    
    try:
        # 如果已经是目标格式，直接返回
        datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
        return time_str
    except ValueError:
        pass
    
    try:
        # 尝试作为 ISO 格式解析并转换
        return utc_convert_beijing(time_str)
    except Exception:
        pass
    
    # 如果无法解析，返回原始字符串
    return time_str

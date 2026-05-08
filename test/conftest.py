#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytest 配置文件 - 提供公共 fixtures
"""

import sys
import os
import pytest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from unittest.mock import MagicMock, patch
import numpy as np

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==================== 测试数据 ====================

@pytest.fixture
def sample_messages():
    """示例聊天消息"""
    return [
        {"role": "user", "content": "我最喜欢蓝色的衣服"},
        {"role": "assistant", "content": "蓝色很好看呢！你还喜欢什么颜色？"},
        {"role": "user", "content": "我也喜欢红色，但是我讨厌绿色"},
        {"role": "assistant", "content": "原来是这样，每个人都有自己喜欢的颜色"},
        {"role": "user", "content": "我的好朋友叫小明，他住在北京"},
        {"role": "assistant", "content": "小明是个好名字，北京是个很棒的城市"},
    ]


@pytest.fixture
def sample_user_history():
    """示例用户聊天历史"""
    return {
        "child_id": "test_child_001",
        "agent_id": "test_agent_001",
        "sessions": [
            {
                "session_id": "session_001",
                "session_start_time": "2025-02-15T10:00:00+08:00",
                "chat_records": [
                    {"role": "user", "content": "我最喜欢奥特曼"},
                    {"role": "assistant", "content": "奥特曼很厉害呢！"},
                    {"role": "user", "content": "我想要一个奥特曼玩具"},
                    {"role": "assistant", "content": "那是个很棒的愿望！"},
                ]
            },
            {
                "session_id": "session_002",
                "session_start_time": "2025-02-15T14:00:00+08:00",
                "chat_records": [
                    {"role": "user", "content": "我今天学会了骑自行车"},
                    {"role": "assistant", "content": "太棒了！你真厉害！"},
                ]
            }
        ]
    }


@pytest.fixture
def sample_memories():
    """示例记忆数据"""
    return [
        {
            "memory_id": "mem_001",
            "memory_content": "孩子非常喜欢奥特曼",
            "memory_type": "preference",
            "child_id": "test_child_001",
            "agent_id": "test_agent_001",
            "metion_count": 3,
            "updated_at": "2025-02-15T10:00:00Z"
        },
        {
            "memory_id": "mem_002",
            "memory_content": "孩子想要一个奥特曼玩具",
            "memory_type": "preference",
            "child_id": "test_child_001",
            "agent_id": "test_agent_001",
            "metion_count": 1,
            "updated_at": "2025-02-15T10:30:00Z"
        },
        {
            "memory_id": "mem_003",
            "memory_content": "孩子学会了骑自行车",
            "memory_type": "ability",
            "child_id": "test_child_001",
            "agent_id": "test_agent_001",
            "metion_count": 1,
            "updated_at": "2025-02-15T14:00:00Z"
        }
    ]


@pytest.fixture
def sample_topic_segments():
    """示例主题分割结果"""
    return [
        [
            {"role": "user", "content": "我最喜欢奥特曼"},
            {"role": "assistant", "content": "奥特曼很厉害呢！"},
            {"role": "user", "content": "我想要一个奥特曼玩具"},
            {"role": "assistant", "content": "那是个很棒的愿望！"},
        ],
        [
            {"role": "user", "content": "我今天学会了骑自行车"},
            {"role": "assistant", "content": "太棒了！你真厉害！"},
        ]
    ]


@pytest.fixture
def time_range():
    """测试时间范围"""
    end_time = datetime.now(ZoneInfo('Asia/Shanghai'))
    start_time = end_time - timedelta(hours=24)
    return start_time, end_time


# ==================== Mock Fixtures ====================

@pytest.fixture
def mock_embedding_client():
    """Mock Embedding 客户端"""
    mock = MagicMock()
    # 返回 768 维的随机向量
    mock.get_embeddings.return_value = np.random.rand(1, 768).astype(np.float32)
    return mock


@pytest.fixture
def mock_es_client():
    """Mock ES 客户端"""
    mock = MagicMock()
    mock.search.return_value = []
    mock.insert_one.return_value = {"_id": "test_id", "result": "created"}
    mock.update_one.return_value = {"_id": "test_id", "result": "updated"}
    return mock


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant 记忆客户端"""
    mock = MagicMock()
    mock.search.return_value = []
    mock.scroll_all.return_value = []
    mock.upsert_one.return_value = None
    mock.set_payload.return_value = None
    mock.overwrite_payload.return_value = None
    mock.delete_one.return_value = None
    mock.get_by_ids.return_value = []
    return mock


@pytest.fixture
def mock_llm_api():
    """Mock LLM API"""
    mock = MagicMock()
    mock.chat.return_value = '''[
        {"content": "孩子非常喜欢奥特曼", "type": "偏好记忆"},
        {"content": "孩子想要一个奥特曼玩具", "type": "偏好记忆"}
    ]'''
    return mock


@pytest.fixture
def mock_whale_api():
    """Mock Whale API"""
    mock = MagicMock()
    mock.get_base_info.return_value = [
        {"child_info": {"child_id": "test_child_001"}, "agent_id": "test_agent_001"},
        {"child_info": {"child_id": "test_child_002"}, "agent_id": "test_agent_002"},
    ]
    mock.query_chat_info.return_value = [
        {
            "session_id": "session_001",
            "request_time": "2025-02-15T10:00:00+08:00",
            "intent": "chat",
            "chat_records_item": {
                "RequestContent": "我最喜欢奥特曼",
                "ResponseContent": "奥特曼很厉害呢！"
            }
        }
    ]
    return mock


# ==================== 配置 Fixtures ====================

@pytest.fixture
def mock_config():
    """Mock 配置"""
    mock = MagicMock()
    mock.embedding.HOST = "localhost"
    mock.embedding.PORT = 50051
    mock.es.MEMORY_INDEX = "test_memory_index"
    mock.qdrant.COLLECTION = "test_memory_collection"
    mock.qdrant.VECTOR_SIZE = 768
    mock.qdrant.DISTANCE = "cosine"
    mock.llm.API_KEY = "test_api_key"
    mock.llm.BASE_URL = "https://api.test.com"
    mock.llm.MODEL = "test-model"
    return mock

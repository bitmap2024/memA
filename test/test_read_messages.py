#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ReadMessages 类的单元测试
使用 mock 模拟 WhaleAPI 的外部依赖
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from collections import defaultdict

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_loader.read_messages import ReadMessages
from config.config import Config


class TestReadMessages:
    """ReadMessages 测试类"""
    
    @pytest.fixture
    def mock_config(self):
        """创建模拟的 Config 对象"""
        config = Mock(spec=Config)
        config.whale = Mock()
        config.whale.HOST = "http://test-host:8000"
        return config
    
    @pytest.fixture
    def mock_whale_api(self):
        """创建模拟的 WhaleAPI 响应"""
        return [
            {
                "child_info": {"child_id": "child_001"},
                "agent_id": "agent_001"
            },
            {
                "child_info": {"child_id": "child_002"},
                "agent_id": "agent_002"
            }
        ]
    
    @pytest.fixture
    def mock_chat_data(self):
        """创建模拟的聊天记录数据"""
        return [
            {
                "session_id": "session_001",
                "request_time": "2026-02-13T10:00:00",
                "intent": "chat",
                "chat_records_item": {
                    "RequestContent": "你好",
                    "ResponseContent": "你好！有什么可以帮助你的吗？"
                }
            },
            {
                "session_id": "session_001",
                "request_time": "2026-02-13T10:00:00",
                "intent": "chat",
                "chat_records_item": {
                    "RequestContent": "今天天气怎么样？",
                    "ResponseContent": "今天天气晴朗，适合外出。"
                }
            },
            {
                "session_id": "session_002",
                "request_time": "2026-02-13T11:00:00",
                "intent": "story",
                "chat_records_item": {
                    "RequestContent": "给我讲个故事",
                    "ResponseContent": "从前有座山..."
                }
            }
        ]
    
    @patch('data_loader.read_messages.WhaleAPI')
    def test_init_success(self, mock_whale_class, mock_config, mock_whale_api):
        """测试 ReadMessages 初始化成功"""
        mock_whale_instance = Mock()
        mock_whale_instance.get_base_info.return_value = mock_whale_api
        mock_whale_class.return_value = mock_whale_instance
        
        reader = ReadMessages(mock_config)
        
        assert len(reader.ids) == 2
        assert reader.ids[0]["child_id"] == "child_001"
        assert reader.ids[0]["agent_id"] == "agent_001"
        assert reader.ids[1]["child_id"] == "child_002"
        assert reader.ids[1]["agent_id"] == "agent_002"
    
    @patch('data_loader.read_messages.WhaleAPI')
    def test_init_empty_response_raises_exception(self, mock_whale_class, mock_config):
        """测试当 get_base_info 返回空列表时抛出异常"""
        mock_whale_instance = Mock()
        mock_whale_instance.get_base_info.return_value = []
        mock_whale_class.return_value = mock_whale_instance
        
        with pytest.raises(Exception) as exc_info:
            ReadMessages(mock_config)
        
        assert "get_user_ids's data is []" in str(exc_info.value)
    
    @patch('data_loader.read_messages.WhaleAPI')
    def test_init_with_invalid_agent_format(self, mock_whale_class, mock_config):
        """测试处理格式错误的 agent 数据"""
        mock_whale_instance = Mock()
        mock_whale_instance.get_base_info.return_value = [
            {"child_info": {"child_id": "child_001"}, "agent_id": "agent_001"},
            {"invalid_format": "data"},  # 格式错误的数据
            {"child_info": {"child_id": "child_003"}, "agent_id": "agent_003"}
        ]
        mock_whale_class.return_value = mock_whale_instance
        
        reader = ReadMessages(mock_config)
        
        # 应该跳过格式错误的数据，只保留有效的
        assert len(reader.ids) == 2
        assert reader.ids[0]["child_id"] == "child_001"
        assert reader.ids[1]["child_id"] == "child_003"
    
    @patch('data_loader.read_messages.WhaleAPI')
    def test_get_single_user_history_success(self, mock_whale_class, mock_config, mock_whale_api, mock_chat_data):
        """测试获取单个用户历史记录成功"""
        mock_whale_instance = Mock()
        mock_whale_instance.get_base_info.return_value = mock_whale_api
        mock_whale_instance.query_chat_info.return_value = mock_chat_data
        mock_whale_class.return_value = mock_whale_instance
        
        reader = ReadMessages(mock_config)
        
        start_time = datetime(2026, 2, 13, 0, 0, 0)
        end_time = datetime(2026, 2, 13, 23, 59, 59)
        
        result = reader.get_single_user_history("child_001", "agent_001", start_time, end_time)
        
        assert result is not None
        assert result["child_id"] == "child_001"
        assert result["agent_id"] == "agent_001"
        assert len(result["sessions"]) == 2  # 两个 session
        
        # 验证 session_001 有 4 条记录 (2 轮对话)
        session_001 = next(s for s in result["sessions"] if s["session_id"] == "session_001")
        assert len(session_001["chat_records"]) == 4
        assert session_001["chat_records"][0]["role"] == "user"
        assert session_001["chat_records"][0]["content"] == "你好"
        assert session_001["chat_records"][1]["role"] == "assistant"
        
        # 验证 session_002 有 2 条记录 (1 轮对话)
        session_002 = next(s for s in result["sessions"] if s["session_id"] == "session_002")
        assert len(session_002["chat_records"]) == 2
    
    @patch('data_loader.read_messages.WhaleAPI')
    def test_get_single_user_history_empty_agent_id(self, mock_whale_class, mock_config, mock_whale_api):
        """测试当 agent_id 为空时返回 None"""
        mock_whale_instance = Mock()
        mock_whale_instance.get_base_info.return_value = mock_whale_api
        mock_whale_class.return_value = mock_whale_instance
        
        reader = ReadMessages(mock_config)
        
        start_time = datetime(2026, 2, 13, 0, 0, 0)
        end_time = datetime(2026, 2, 13, 23, 59, 59)
        
        result = reader.get_single_user_history("child_001", "", start_time, end_time)
        
        assert result is None
    
    @patch('data_loader.read_messages.WhaleAPI')
    def test_get_single_user_history_no_chat_data(self, mock_whale_class, mock_config, mock_whale_api):
        """测试当没有聊天数据时返回 None"""
        mock_whale_instance = Mock()
        mock_whale_instance.get_base_info.return_value = mock_whale_api
        mock_whale_instance.query_chat_info.return_value = []
        mock_whale_class.return_value = mock_whale_instance
        
        reader = ReadMessages(mock_config)
        
        start_time = datetime(2026, 2, 13, 0, 0, 0)
        end_time = datetime(2026, 2, 13, 23, 59, 59)
        
        result = reader.get_single_user_history("child_001", "agent_001", start_time, end_time)
        
        assert result is None
    
    @patch('data_loader.read_messages.WhaleAPI')
    def test_get_single_user_history_filters_by_intent(self, mock_whale_class, mock_config, mock_whale_api):
        """测试只从指定 intent 类型中提取记忆"""
        mock_whale_instance = Mock()
        mock_whale_instance.get_base_info.return_value = mock_whale_api
        mock_whale_instance.query_chat_info.return_value = [
            {
                "session_id": "session_001",
                "request_time": "2026-02-13T10:00:00",
                "intent": "chat",  # 应该被包含
                "chat_records_item": {
                    "RequestContent": "你好",
                    "ResponseContent": "你好！"
                }
            },
            {
                "session_id": "session_001",
                "request_time": "2026-02-13T10:00:00",
                "intent": "unknown_intent",  # 应该被过滤
                "chat_records_item": {
                    "RequestContent": "这条不应该出现",
                    "ResponseContent": "这条也不应该出现"
                }
            },
            {
                "session_id": "session_001",
                "request_time": "2026-02-13T10:00:00",
                "intent": "music",  # 应该被包含
                "chat_records_item": {
                    "RequestContent": "播放音乐",
                    "ResponseContent": "正在播放..."
                }
            }
        ]
        mock_whale_class.return_value = mock_whale_instance
        
        reader = ReadMessages(mock_config)
        
        start_time = datetime(2026, 2, 13, 0, 0, 0)
        end_time = datetime(2026, 2, 13, 23, 59, 59)
        
        result = reader.get_single_user_history("child_001", "agent_001", start_time, end_time)
        
        assert result is not None
        session = result["sessions"][0]
        # 应该只有 4 条记录 (2 轮有效对话，过滤掉了 unknown_intent)
        assert len(session["chat_records"]) == 4
        assert "这条不应该出现" not in [r["content"] for r in session["chat_records"]]
    
    @patch('data_loader.read_messages.WhaleAPI')
    def test_get_single_user_history_non_list_response(self, mock_whale_class, mock_config, mock_whale_api):
        """测试当 query_chat_info 返回非 list 时抛出异常"""
        mock_whale_instance = Mock()
        mock_whale_instance.get_base_info.return_value = mock_whale_api
        mock_whale_instance.query_chat_info.return_value = {"invalid": "response"}  # 非 list
        mock_whale_class.return_value = mock_whale_instance
        
        reader = ReadMessages(mock_config)
        
        start_time = datetime(2026, 2, 13, 0, 0, 0)
        end_time = datetime(2026, 2, 13, 23, 59, 59)
        
        # 应该返回 None（因为异常被捕获）
        result = reader.get_single_user_history("child_001", "agent_001", start_time, end_time)
        assert result is None
    
    @patch('data_loader.read_messages.WhaleAPI')
    def test_get_user_history_batch(self, mock_whale_class, mock_config, mock_whale_api, mock_chat_data):
        """测试批量获取所有用户历史记录"""
        mock_whale_instance = Mock()
        mock_whale_instance.get_base_info.return_value = mock_whale_api
        mock_whale_instance.query_chat_info.return_value = mock_chat_data
        mock_whale_class.return_value = mock_whale_instance
        
        reader = ReadMessages(mock_config)
        
        start_time = datetime(2026, 2, 13, 0, 0, 0)
        end_time = datetime(2026, 2, 13, 23, 59, 59)
        
        results = reader.get_user_history(start_time, end_time)
        
        # 应该有 2 个用户的历史记录
        assert len(results) == 2
        assert results[0]["child_id"] == "child_001"
        assert results[1]["child_id"] == "child_002"
    
    @patch('data_loader.read_messages.WhaleAPI')
    def test_get_user_history_partial_failure(self, mock_whale_class, mock_config, mock_whale_api, mock_chat_data):
        """测试当部分用户查询失败时，仍能返回成功的结果"""
        mock_whale_instance = Mock()
        mock_whale_instance.get_base_info.return_value = mock_whale_api
        
        # 第一个用户成功，第二个用户返回空
        def side_effect_query(*args, **kwargs):
            if kwargs.get('child_id') == 'child_001':
                return mock_chat_data
            return []
        
        mock_whale_instance.query_chat_info.side_effect = side_effect_query
        mock_whale_class.return_value = mock_whale_instance
        
        reader = ReadMessages(mock_config)
        
        start_time = datetime(2026, 2, 13, 0, 0, 0)
        end_time = datetime(2026, 2, 13, 23, 59, 59)
        
        results = reader.get_user_history(start_time, end_time)
        
        # 只有 1 个用户有数据
        assert len(results) == 1
        assert results[0]["child_id"] == "child_001"
    
    @patch('data_loader.read_messages.WhaleAPI')
    def test_get_single_user_history_empty_sessions_filtered(self, mock_whale_class, mock_config, mock_whale_api):
        """测试没有聊天记录的会话会被过滤掉"""
        mock_whale_instance = Mock()
        mock_whale_instance.get_base_info.return_value = mock_whale_api
        mock_whale_instance.query_chat_info.return_value = [
            {
                "session_id": "session_001",
                "request_time": "2026-02-13T10:00:00",
                "intent": "invalid_intent",  # 会被过滤，导致会话为空
                "chat_records_item": {
                    "RequestContent": "test",
                    "ResponseContent": "test"
                }
            }
        ]
        mock_whale_class.return_value = mock_whale_instance
        
        reader = ReadMessages(mock_config)
        
        start_time = datetime(2026, 2, 13, 0, 0, 0)
        end_time = datetime(2026, 2, 13, 23, 59, 59)
        
        result = reader.get_single_user_history("child_001", "agent_001", start_time, end_time)
        
        # 所有会话都被过滤掉了，应该返回 None
        assert result is None


class TestReadMessagesIntentFiltering:
    """测试 intent 过滤逻辑"""
    
    @pytest.fixture
    def mock_config(self):
        config = Mock(spec=Config)
        config.whale = Mock()
        config.whale.HOST = "http://test-host:8000"
        return config
    
    @pytest.fixture
    def mock_whale_api(self):
        return [{"child_info": {"child_id": "child_001"}, "agent_id": "agent_001"}]
    
    @pytest.mark.parametrize("intent", [
        "chat", "story", "music", "holiday", "weather", "time", "control", "profile"
    ])
    @patch('data_loader.read_messages.WhaleAPI')
    def test_valid_intents_are_included(self, mock_whale_class, mock_config, mock_whale_api, intent):
        """测试所有有效的 intent 类型都会被包含"""
        mock_whale_instance = Mock()
        mock_whale_instance.get_base_info.return_value = mock_whale_api
        mock_whale_instance.query_chat_info.return_value = [
            {
                "session_id": "session_001",
                "request_time": "2026-02-13T10:00:00",
                "intent": intent,
                "chat_records_item": {
                    "RequestContent": f"测试 {intent}",
                    "ResponseContent": f"响应 {intent}"
                }
            }
        ]
        mock_whale_class.return_value = mock_whale_instance
        
        reader = ReadMessages(mock_config)
        
        start_time = datetime(2026, 2, 13, 0, 0, 0)
        end_time = datetime(2026, 2, 13, 23, 59, 59)
        
        result = reader.get_single_user_history("child_001", "agent_001", start_time, end_time)
        
        assert result is not None
        assert len(result["sessions"]) == 1
        assert len(result["sessions"][0]["chat_records"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

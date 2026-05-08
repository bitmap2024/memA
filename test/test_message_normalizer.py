#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试消息标准化模块 (data_loader/message_normalizer.py)
"""

import pytest
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader.message_normalizer import MessageNormalizer


class TestMessageNormalizer:
    """测试 MessageNormalizer 类"""
    
    @pytest.fixture
    def normalizer(self):
        """创建 MessageNormalizer 实例"""
        return MessageNormalizer()
    
    def test_init(self):
        """测试初始化"""
        normalizer = MessageNormalizer(offset_ms=500)
        assert normalizer.offset == timedelta(milliseconds=500)
        assert normalizer.last_timestamp_map == {}
    
    def test_normalize_single_message(self, normalizer):
        """测试单条消息标准化"""
        message = {
            "role": "user",
            "content": "你好",
            "time_stamp": "2025/02/15 (Sat) 10:00"
        }
        
        result = normalizer.normalize_messages(message)
        
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "你好"
        assert "session_time" in result[0]
        assert "weekday" in result[0]
    
    def test_normalize_multiple_messages(self, normalizer):
        """测试多条消息标准化"""
        messages = [
            {"role": "user", "content": "你好", "time_stamp": "2025/02/15 (Sat) 10:00"},
            {"role": "assistant", "content": "你好！", "time_stamp": "2025/02/15 (Sat) 10:00"},
        ]
        
        result = normalizer.normalize_messages(messages)
        
        assert len(result) == 2
        assert result[0]["time_stamp"] != result[1]["time_stamp"]  # 时间应该递增
    
    def test_parse_session_timestamp_standard(self, normalizer):
        """测试标准格式时间戳解析"""
        # 格式: 2025/02/15 (Sat) 10:00
        base_dt, weekday = normalizer._parse_session_timestamp("2025/02/15 (Sat) 10:00")
        
        assert base_dt.year == 2025
        assert base_dt.month == 2
        assert base_dt.day == 15
        assert base_dt.hour == 10
        assert base_dt.minute == 0
        assert weekday == "Sat"
    
    def test_parse_session_timestamp_with_dash(self, normalizer):
        """测试使用短横线分隔的时间戳"""
        # 格式: 2025-02-15 (Sat) 10:00
        base_dt, weekday = normalizer._parse_session_timestamp("2025-02-15 (Sat) 10:00")
        
        assert base_dt.year == 2025
        assert base_dt.month == 2
        assert base_dt.day == 15
    
    def test_parse_session_timestamp_with_seconds(self, normalizer):
        """测试带秒的时间戳"""
        # 格式: 2025/02/15 (Sat) 10:00:30
        base_dt, weekday = normalizer._parse_session_timestamp("2025/02/15 (Sat) 10:00:30")
        
        assert base_dt.second == 30
    
    def test_parse_session_timestamp_iso_format(self, normalizer):
        """测试 ISO 格式时间戳"""
        base_dt, weekday = normalizer._parse_session_timestamp("2025-02-15T10:00:00")
        
        assert base_dt.year == 2025
        assert base_dt.month == 2
        assert base_dt.day == 15
    
    def test_parse_session_timestamp_invalid(self, normalizer):
        """测试无效时间戳"""
        with pytest.raises(ValueError):
            normalizer._parse_session_timestamp("invalid timestamp")
    
    def test_timestamp_incrementing(self, normalizer):
        """测试时间戳递增"""
        messages = [
            {"role": "user", "content": "消息1", "time_stamp": "2025/02/15 (Sat) 10:00"},
            {"role": "assistant", "content": "消息2", "time_stamp": "2025/02/15 (Sat) 10:00"},
            {"role": "user", "content": "消息3", "time_stamp": "2025/02/15 (Sat) 10:00"},
        ]
        
        result = normalizer.normalize_messages(messages)
        
        # 解析时间戳并比较
        times = [datetime.fromisoformat(r["time_stamp"]) for r in result]
        
        assert times[1] > times[0]
        assert times[2] > times[1]
    
    def test_different_sessions(self, normalizer):
        """测试不同会话的时间戳"""
        messages = [
            {"role": "user", "content": "会话1消息1", "time_stamp": "2025/02/15 (Sat) 10:00"},
            {"role": "user", "content": "会话2消息1", "time_stamp": "2025/02/15 (Sat) 14:00"},
        ]
        
        result = normalizer.normalize_messages(messages)
        
        # 不同会话应该有不同的基础时间
        time1 = datetime.fromisoformat(result[0]["time_stamp"])
        time2 = datetime.fromisoformat(result[1]["time_stamp"])
        
        assert time2 > time1
    
    def test_preserve_original_fields(self, normalizer):
        """测试保留原始字段"""
        message = {
            "role": "user",
            "content": "你好",
            "time_stamp": "2025/02/15 (Sat) 10:00",
            "custom_field": "custom_value"
        }
        
        result = normalizer.normalize_messages(message)
        
        assert result[0]["custom_field"] == "custom_value"
    
    def test_missing_timestamp_error(self, normalizer):
        """测试缺少时间戳的错误"""
        message = {"role": "user", "content": "你好"}
        
        with pytest.raises(ValueError) as exc_info:
            normalizer.normalize_messages(message)
        
        assert "time_stamp" in str(exc_info.value)
    
    def test_string_input_error(self, normalizer):
        """测试字符串输入的错误"""
        with pytest.raises(ValueError) as exc_info:
            normalizer.normalize_messages("这是一条消息")
        
        assert "dict or list[dict]" in str(exc_info.value)
    
    def test_invalid_list_item_error(self, normalizer):
        """测试列表中包含非字典元素的错误"""
        messages = [
            {"role": "user", "content": "消息1", "time_stamp": "2025/02/15 (Sat) 10:00"},
            "这不是一个字典"
        ]
        
        with pytest.raises(ValueError) as exc_info:
            normalizer.normalize_messages(messages)
        
        assert "dict" in str(exc_info.value)
    
    def test_deep_copy(self, normalizer):
        """测试深拷贝不影响原始数据"""
        message = {
            "role": "user",
            "content": "你好",
            "time_stamp": "2025/02/15 (Sat) 10:00"
        }
        original_time_stamp = message["time_stamp"]
        
        result = normalizer.normalize_messages(message)
        
        # 原始消息不应该被修改
        assert message["time_stamp"] == original_time_stamp
        # 结果应该有新的 time_stamp
        assert result[0]["time_stamp"] != original_time_stamp


class TestMessageNormalizerEdgeCases:
    """边界情况测试"""
    
    def test_empty_content(self):
        """测试空内容"""
        normalizer = MessageNormalizer()
        message = {
            "role": "user",
            "content": "",
            "time_stamp": "2025/02/15 (Sat) 10:00"
        }
        
        result = normalizer.normalize_messages(message)
        
        assert result[0]["content"] == ""
    
    def test_very_long_content(self):
        """测试超长内容"""
        normalizer = MessageNormalizer()
        long_content = "x" * 10000
        message = {
            "role": "user",
            "content": long_content,
            "time_stamp": "2025/02/15 (Sat) 10:00"
        }
        
        result = normalizer.normalize_messages(message)
        
        assert len(result[0]["content"]) == 10000
    
    def test_special_characters_in_content(self):
        """测试特殊字符"""
        normalizer = MessageNormalizer()
        message = {
            "role": "user",
            "content": "你好！🎉 \"quote\" <tag> & special",
            "time_stamp": "2025/02/15 (Sat) 10:00"
        }
        
        result = normalizer.normalize_messages(message)
        
        assert "🎉" in result[0]["content"]
        assert "\"quote\"" in result[0]["content"]
    
    def test_custom_offset(self):
        """测试自定义偏移量"""
        normalizer = MessageNormalizer(offset_ms=5000)  # 5秒
        messages = [
            {"role": "user", "content": "消息1", "time_stamp": "2025/02/15 (Sat) 10:00"},
            {"role": "assistant", "content": "消息2", "time_stamp": "2025/02/15 (Sat) 10:00"},
        ]
        
        result = normalizer.normalize_messages(messages)
        
        time1 = datetime.fromisoformat(result[0]["time_stamp"])
        time2 = datetime.fromisoformat(result[1]["time_stamp"])
        
        # 差异应该是 5 秒
        diff = (time2 - time1).total_seconds()
        assert diff == 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试文本压缩模块 (extract_mem/text_compressor.py)
"""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTextCompressor:
    """测试 TextCompressor 类"""
    
    @pytest.fixture
    def mock_compressor(self):
        """创建 Mock 的压缩器"""
        with patch('extract_mem.text_compressor.PromptCompressor') as mock:
            mock_instance = MagicMock()
            mock_instance.compress_prompt_llmlingua2.return_value = {
                "compressed_prompt": "压缩后的文本",
                "compressed_tokens": 50,
                "origin_tokens": 100,
                "ratio": 0.5,
                "fn_labeled_original_prompt": "word1 1\t\t|\t\tword2 0\t\t|\t\tword3 1"
            }
            mock.return_value = mock_instance
            yield mock_instance
    
    def test_compress_basic(self, mock_compressor):
        """测试基本压缩功能"""
        from extract_mem.text_compressor import TextCompressor
        
        with patch('extract_mem.text_compressor.PromptCompressor', return_value=mock_compressor):
            compressor = TextCompressor(model_path="/fake/path")
            result = compressor.compress("这是一段很长的测试文本，需要被压缩")
            
            assert "compressed_prompt" in result
            assert "compressed_tokens" in result
            assert result["compressed_tokens"] <= result["origin_tokens"]
    
    def test_compress_with_rate(self, mock_compressor):
        """测试不同压缩率"""
        from extract_mem.text_compressor import TextCompressor
        
        with patch('extract_mem.text_compressor.PromptCompressor', return_value=mock_compressor):
            compressor = TextCompressor(model_path="/fake/path")
            
            # 测试不同压缩率
            for rate in [0.3, 0.5, 0.7]:
                result = compressor.compress("测试文本", rate=rate)
                mock_compressor.compress_prompt_llmlingua2.assert_called()
    
    def test_get_annotated_results(self, mock_compressor):
        """测试获取标注结果"""
        from extract_mem.text_compressor import TextCompressor
        
        with patch('extract_mem.text_compressor.PromptCompressor', return_value=mock_compressor):
            compressor = TextCompressor(model_path="/fake/path")
            
            results = {
                "fn_labeled_original_prompt": "保留 1\t\t|\t\t删除 0\t\t|\t\t保留2 1"
            }
            annotated = compressor.get_annotated_results(results)
            
            assert len(annotated) == 3
            assert annotated[0] == ("保留", "+")
            assert annotated[1] == ("删除", "-")
            assert annotated[2] == ("保留2", "+")
    
    def test_compress_and_annotate_messages(self, mock_compressor):
        """测试压缩消息列表"""
        from extract_mem.text_compressor import TextCompressor
        
        with patch('extract_mem.text_compressor.PromptCompressor', return_value=mock_compressor):
            compressor = TextCompressor(model_path="/fake/path")
            compressor.max_tokens = 100  # 设置最大 token 数
            
            messages = [
                {"role": "user", "content": "这是用户的消息"},
                {"role": "assistant", "content": "这是助手的回复"}
            ]
            
            result = compressor.compress_and_annotate(messages, rate=0.5)
            
            # 验证消息被处理
            assert len(result) == 2
            assert "content" in result[0]
            assert "content" in result[1]
    
    def test_compress_empty_content(self, mock_compressor):
        """测试空内容处理"""
        from extract_mem.text_compressor import TextCompressor
        
        with patch('extract_mem.text_compressor.PromptCompressor', return_value=mock_compressor):
            compressor = TextCompressor(model_path="/fake/path")
            compressor.max_tokens = 100
            
            messages = [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": "   "},
                {"role": "user", "content": "有效内容"}
            ]
            
            result = compressor.compress_and_annotate(messages, rate=0.5)
            
            # 空内容应该被跳过
            assert len(result) == 3


class TestTextCompressorIntegration:
    """集成测试（需要实际模型）"""
    
    @pytest.mark.skip(reason="需要实际模型文件")
    def test_real_compression(self):
        """测试实际压缩（需要模型）"""
        from extract_mem.text_compressor import TextCompressor
        
        model_path = "/root/chendong/hf_models/microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
        compressor = TextCompressor(model_path=model_path)
        
        text = """
        今天天气真好，我和妈妈一起去公园玩。
        我们看到了很多漂亮的花，还有小鸟在唱歌。
        我最喜欢红色的玫瑰花，妈妈说下次可以买一束回家。
        """
        
        result = compressor.compress(text, rate=0.5)
        
        assert len(result["compressed_prompt"]) < len(text)
        assert result["ratio"] <= 0.6  # 压缩率应该接近 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

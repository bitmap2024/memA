#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 Embedding 客户端模块 (api/emb_api.py)
"""

import pytest
import sys
import os
import numpy as np
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEmbeddingClient:
    """测试 EmbeddingClient 类"""
    
    @pytest.fixture
    def mock_grpc_channel(self):
        """Mock gRPC channel"""
        with patch('api.emb_api.grpc.insecure_channel') as mock_channel:
            mock_channel.return_value = MagicMock()
            yield mock_channel
    
    @pytest.fixture
    def mock_stub(self):
        """Mock gRPC stub"""
        mock = MagicMock()
        mock_response = MagicMock()
        mock_response.embeddings = np.random.rand(768).tolist()
        mock_response.embedding_dim = 768
        mock.Encode.return_value = mock_response
        return mock
    
    def test_client_init(self, mock_grpc_channel):
        """测试客户端初始化"""
        from api.emb_api import EmbeddingClient
        
        client = EmbeddingClient(host="localhost", port=50051)
        
        assert client.target == "localhost:50051"
        assert client.timeout == 30.0
        assert client.max_retries == 3
    
    def test_client_connect(self, mock_grpc_channel, mock_stub):
        """测试客户端连接"""
        with patch('api.emb_api.embedding_service_pb2_grpc.EmbeddingServiceStub', return_value=mock_stub):
            from api.emb_api import EmbeddingClient
            
            client = EmbeddingClient(host="localhost", port=50051)
            client.connect()
            
            assert client.channel is not None
            assert client.stub is not None
    
    def test_client_close(self, mock_grpc_channel, mock_stub):
        """测试客户端关闭"""
        with patch('api.emb_api.embedding_service_pb2_grpc.EmbeddingServiceStub', return_value=mock_stub):
            from api.emb_api import EmbeddingClient
            
            client = EmbeddingClient(host="localhost", port=50051)
            client.connect()
            client.close()
            
            assert client.channel is None
            assert client.stub is None
    
    def test_get_embeddings(self, mock_grpc_channel, mock_stub):
        """测试获取 embeddings"""
        with patch('api.emb_api.embedding_service_pb2_grpc.EmbeddingServiceStub', return_value=mock_stub):
            from api.emb_api import EmbeddingClient
            
            client = EmbeddingClient(host="localhost", port=50051)
            client.connect()
            
            sentences = ["测试句子1", "测试句子2"]
            
            # Mock 返回值
            mock_response = MagicMock()
            mock_response.embeddings = np.random.rand(2 * 768).tolist()
            mock_response.embedding_dim = 768
            mock_stub.Encode.return_value = mock_response
            
            result = client.get_embeddings(sentences)
            
            assert isinstance(result, np.ndarray)
    
    def test_get_embeddings_batch(self, mock_grpc_channel, mock_stub):
        """测试批量获取 embeddings"""
        with patch('api.emb_api.embedding_service_pb2_grpc.EmbeddingServiceStub', return_value=mock_stub):
            from api.emb_api import EmbeddingClient
            
            client = EmbeddingClient(host="localhost", port=50051)
            client.connect()
            
            # 150 个句子，测试批处理
            sentences = [f"测试句子{i}" for i in range(150)]
            
            mock_response = MagicMock()
            mock_response.embeddings = np.random.rand(100 * 768).tolist()
            mock_response.embedding_dim = 768
            mock_stub.Encode.return_value = mock_response
            
            result = client.get_embeddings(sentences, batch_size=100)
            
            # 应该调用两次（100 + 50）
            assert mock_stub.Encode.call_count >= 1


class TestEmbeddingClientPool:
    """测试 EmbeddingClientPool 类"""
    
    @pytest.fixture
    def mock_client(self):
        """Mock 单个客户端"""
        mock = MagicMock()
        mock.get_embeddings.return_value = np.random.rand(1, 768)
        return mock
    
    def test_pool_init(self):
        """测试池初始化"""
        with patch('api.emb_api.EmbeddingClient') as mock_client_class:
            mock_client_class.return_value = MagicMock()
            
            from api.emb_api import EmbeddingClientPool
            
            pool = EmbeddingClientPool(host="localhost", port=50051, pool_size=3)
            
            assert len(pool.clients) == 3
    
    def test_pool_get_client(self):
        """测试获取客户端"""
        with patch('api.emb_api.EmbeddingClient') as mock_client_class:
            mock_clients = [MagicMock() for _ in range(3)]
            mock_client_class.side_effect = mock_clients
            
            from api.emb_api import EmbeddingClientPool
            
            pool = EmbeddingClientPool(host="localhost", port=50051, pool_size=3)
            
            # 获取客户端应该轮询
            client1 = pool.get_client()
            client2 = pool.get_client()
            client3 = pool.get_client()
            client4 = pool.get_client()
            
            # 第4个应该回到第1个
            assert client1 == mock_clients[0]
            assert client2 == mock_clients[1]
            assert client3 == mock_clients[2]
            assert client4 == mock_clients[0]
    
    def test_pool_close_all(self):
        """测试关闭所有客户端"""
        with patch('api.emb_api.EmbeddingClient') as mock_client_class:
            mock_clients = [MagicMock() for _ in range(3)]
            mock_client_class.side_effect = mock_clients
            
            from api.emb_api import EmbeddingClientPool
            
            pool = EmbeddingClientPool(host="localhost", port=50051, pool_size=3)
            pool.close_all()
            
            # 验证所有客户端都被关闭
            for client in mock_clients:
                client.close.assert_called_once()
    
    def test_pool_thread_safety(self):
        """测试池的线程安全性"""
        import threading
        
        with patch('api.emb_api.EmbeddingClient') as mock_client_class:
            mock_client_class.return_value = MagicMock()
            
            from api.emb_api import EmbeddingClientPool
            
            pool = EmbeddingClientPool(host="localhost", port=50051, pool_size=3)
            
            results = []
            
            def get_client():
                client = pool.get_client()
                results.append(client)
            
            threads = [threading.Thread(target=get_client) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            
            # 所有线程都应该成功获取客户端
            assert len(results) == 10


class TestRetryDecorator:
    """测试重试装饰器"""
    
    def test_retry_on_success(self):
        """测试成功时不重试"""
        from api.emb_api import retry_on_failure
        
        call_count = 0
        
        @retry_on_failure(max_retries=3)
        def success_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = success_func()
        
        assert result == "success"
        assert call_count == 1
    
    def test_retry_on_failure_then_success(self):
        """测试失败后重试成功"""
        import grpc
        from api.emb_api import retry_on_failure
        
        call_count = 0
        
        @retry_on_failure(max_retries=3, initial_delay=0.01)
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                error = grpc.RpcError()
                error.code = lambda: grpc.StatusCode.UNAVAILABLE
                raise error
            return "success"
        
        result = flaky_func()
        
        assert result == "success"
        assert call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

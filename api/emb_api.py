#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : embedding_client.py


import sys
import grpc
import time
import threading
import numpy as np
from functools import wraps
from typing import List, Optional
from loguru import logger
from protos import embedding_service_pb2_grpc, embedding_service_pb2

def retry_on_failure(max_retries=3, initial_delay=1, backoff_multiplier=2.0):

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except grpc.RpcError as e:
                    if e.code() in [
                        grpc.StatusCode.UNAVAILABLE,
                        grpc.StatusCode.DEADLINE_EXCEEDED,
                        grpc.StatusCode.RESOURCE_EXHAUSTED,
                    ]:
                        logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                        time.sleep(delay)
                        delay *= backoff_multiplier
                    else:
                        raise
            raise RuntimeError("Max retries exceeded")
        return wrapper
    return decorator


class EmbeddingClient:
    """
    Synchronous client for the embedding service.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 50051,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.target = f"{host}:{port}"
        self.timeout = timeout
        self.max_retries = max_retries
        self.channel = None
        self.stub = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def connect(self):
        """Establish connection to the server."""
        if self.channel is None:
            options = [
                ("grpc.max_send_message_length", 1000 * 1024 * 1024),  # 1000MB
                ("grpc.max_receive_message_length", 1000 * 1024 * 1024),  # 1000MB
                ("grpc.keepalive_time_ms", 30000),  # 30 seconds
                ("grpc.keepalive_timeout_ms", 10000),  # 10 seconds
                ("grpc.keepalive_permit_without_calls", True),
                ("grpc.http2.max_pings_without_data", 0),
                ("grpc.http2.min_time_between_pings_ms", 10000),  # 10 seconds
                ("grpc.http2.min_ping_interval_without_data_ms", 5000),  # 5 seconds
            ]
            self.channel = grpc.insecure_channel(self.target, options=options)
            self.stub = embedding_service_pb2_grpc.EmbeddingServiceStub(self.channel)

    def close(self):
        """Close the connection."""
        if self.channel is not None:
            self.channel.close()
            self.channel = None
            self.stub = None

    @retry_on_failure(max_retries=3, initial_delay=1, backoff_multiplier=2.0)
    def get_embeddings(
        self, sentences: List[str], timeout: Optional[float] = None, batch_size: int = 100
    ) -> np.ndarray:
        """
        Get embeddings for a list of sentences.

        Args:
            sentences: List of sentences to encode
            timeout: Optional timeout override
            batch_size: Batch size, default 100
        Returns:
            numpy.ndarray: Array of embeddings
        """
        if not self.stub:
            self.connect()

        try:
            n = len(sentences)
            embs = []
            for i in range(0, n, batch_size):
                sens = sentences[i:i+batch_size]
                request = embedding_service_pb2.EncodeRequest(sentences=sens)
                response = self.stub.Encode(request, timeout=timeout or self.timeout)
                embeddings = np.array(response.embeddings)
                embeddings = embeddings.reshape(-1, response.embedding_dim)
                # 将当前批次的 embeddings 添加到 embs 列表中
                embs.append(embeddings)
            final_embeddings = np.concatenate(embs, axis=0)
            print(final_embeddings.shape)
            return final_embeddings

        except Exception as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            raise


class EmbeddingClientPool:
    """
    Pool of embedding clients for handling concurrent requests.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 50051,
        pool_size: int = 5,
        **client_kwargs,
    ):
        self.clients = [
            EmbeddingClient(host, port, **client_kwargs) for _ in range(pool_size)
        ]
        self.current = 0
        self._lock = threading.Lock()

    def get_client(self) -> EmbeddingClient:
        """Get next available client from the pool."""
        with self._lock:
            client = self.clients[self.current]
            self.current = (self.current + 1) % len(self.clients)
            return client

    def close_all(self):
        """Close all clients in the pool."""
        for client in self.clients:
            client.close()

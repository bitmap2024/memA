#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Memory gRPC Server - 记忆系统 gRPC 服务入口
通过 gRPC 接口暴露 MemoryService 的记忆检索能力
"""

import os
import sys
from concurrent import futures
import grpc
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'protos'))

from protos import mem_pb2
from protos import mem_pb2_grpc
from config.config import Config
from services.retrieval_memory_service import RetrievalMemoryService


class MemoryServiceServicer(mem_pb2_grpc.MemoryServiceServicer):
    """
    gRPC 服务实现 - 封装 MemoryService 的记忆检索能力
    """
    
    def __init__(self, memory_service: RetrievalMemoryService):
        self.memory_service = memory_service
        logger.info("MemoryServiceServicer 初始化完成")
    
    def GetUserAllMemory(self, request, context):
        """
        获取指定用户的所有记忆（不限 agent）
        """
        logger.info(f"GetChildAllMemory: child_id={request.child_id}, child_name={request.child_name}")
        
        try:
            memories = self.memory_service.get_all_memory(
                child_id=request.child_id,
                agent_id=None
            )
            
            return self._build_response(memories)
        except Exception as e:
            logger.error(f"GetChildAllMemory 失败: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return mem_pb2.MemResponse(code=-1, message=str(e), data=[])
    
    def GetRelateMemory(self, request, context):
        """
        根据 query 检索相关记忆
        """
        query = request.query if request.HasField('query') else ""
        agent_id = request.agent_id if request.HasField('agent_id') else None
        limit = request.limit if request.HasField('limit') else 10
        intent = request.intent if request.HasField('intent') else None
        
        logger.info(f"GetRelateMemory: child_id={request.child_id}, child_name={request.child_name}, agent_id={agent_id}, intent={intent}, query={repr(query[:50])}")
        
        try:
            if not query:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("query is required")
                return mem_pb2.MemResponse(code=-1, message="query is required", data=[])
            
            memory_contents = self.memory_service.get_relate_memory(
                child_id=request.child_id,
                agent_id=agent_id,
                query=query,
                top_k=limit
            )
            
            return mem_pb2.MemResponse(code=0, message="success", data=memory_contents)
        except Exception as e:
            logger.error(f"GetRelateMemory 失败: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return mem_pb2.MemResponse(code=-1, message=str(e), data=[])
    
    def _build_response(self, memories: list, code: int = 0, message: str = "success") -> mem_pb2.MemResponse:
        """
        构建统一的 MemoryResponse
        """
        data = []
        for m in memories:
            if isinstance(m, dict):
                data.append(m.get("memory_content", ""))
            elif isinstance(m, str):
                data.append(m)
        
        return mem_pb2.MemResponse(code=code, message=message, data=data)


def serve():
    """
    启动 gRPC 服务器
    """
    host = os.getenv("SERVICE_HOST", "0.0.0.0")
    port = os.getenv("SERVICE_PORT", "51666")
    max_workers = int(os.getenv("SERVICE_WORKERS", "8"))
    
    logger.info("=" * 50)
    logger.info("初始化 Memory gRPC 服务...")
    logger.info(f"host: {host}")
    logger.info(f"port: {port}")
    logger.info(f"max_workers: {max_workers}")
    logger.info("=" * 50)
    
    memory_service = RetrievalMemoryService(config=Config)
    logger.info("RetrievalMemoryService 初始化完成")
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    
    servicer = MemoryServiceServicer(memory_service)
    mem_pb2_grpc.add_MemoryServiceServicer_to_server(servicer, server)
    
    server_address = f"{host}:{port}"
    server.add_insecure_port(server_address)
    
    server.start()
    logger.info("=" * 50)
    logger.info(f"Memory gRPC 服务已启动")
    logger.info(f"监听地址: {server_address}")
    logger.info(f"工作线程数: {max_workers}")
    logger.info("=" * 50)
# ssh -R 7897:127.0.0.1:7890 -p 47794 root@connect.bjb2.seetacloud.com
# export https_proxy=http://127.0.0.1:7897
# export http_proxy=http://127.0.0.1:7897
  
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭服务...")
        server.stop(grace=5)
        logger.info("服务已关闭")


if __name__ == "__main__":
    serve()

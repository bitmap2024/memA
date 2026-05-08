#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : config.py

"""
配置管理模块
支持从环境变量读取配置（可通过 start.sh 导出环境变量）
"""

import os
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field


class ServiceConfig(BaseSettings):
    """服务配置"""
    
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    DEBUG: bool = False
    
    class Config:
        env_prefix = "SERVICE_"


class LLMConfig(BaseSettings):
    """LLM 配置"""
    API_KEY: str = "sk-b7d48e7fc9ef44b4b33c80470402d52a"
    BASE_URL: str = "https://api.deepseek.com"
    MODEL: str = "deepseek-chat"
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 2048
    TOP_P: float = 0.95
    STREAM: bool = False
    TIMEOUT: int = 60
    
    class Config:
        env_prefix = "LLM_"

class QdrantConfig(BaseSettings):
    """Qdrant 配置"""

    URL: str = "http://localhost:6333"
    API_KEY: str = ""
    COLLECTION: str = "light_memory_rag"
    VECTOR_SIZE: int = 1024
    DISTANCE: str = "cosine"
    TIMEOUT: int = 30

    class Config:
        env_prefix = "QDRANT_"


class EmbeddingConfig(BaseSettings):
    """Embedding 服务配置"""
    HOST: str = "172.16.2.124"
    PORT: int = 80
    MODEL: str = "bge-m3"
    
    @property
    def URL(self) -> str:
        return f"http://{self.HOST}:{self.PORT}"
    
    class Config:
        env_prefix = "EMBEDDING_"



class CompressorConfig(BaseSettings):
    """Compressor 配置"""
    
    MODEL_PATH: str = "/root/chendong/hf_models/microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
    CONFIGS: Dict[str, Any] = {}
    
    class Config:
        env_prefix = "PRE_COMPRESSOR_"




class Config:
    """全局配置聚合类"""
    
    service = ServiceConfig()
    llm = LLMConfig()
    es = ESConfig()
    qdrant = QdrantConfig()
    embedding = EmbeddingConfig()
    memory = MemoryConfig()
    pre_compressor = PreCompressorConfig()
    whale = WhaleConfig()
    
    @classmethod
    def reload(cls):
        """重新加载配置（从环境变量）"""
        cls.service = ServiceConfig()
        cls.llm = LLMConfig()
        cls.es = ESConfig()
        cls.qdrant = QdrantConfig()
        cls.embedding = EmbeddingConfig()
        cls.memory = MemoryConfig()
    
    @classmethod
    def print_config(cls):
        """打印当前配置（隐藏敏感信息）"""
        def mask_secret(value: str, show_chars: int = 4) -> str:
            if not value or len(value) <= show_chars:
                return "***"
            return value[:show_chars] + "***"
        
        print("=" * 50)
        print("Current Configuration")
        print("=" * 50)
        
        print("\n[Service]")
        print(f"  HOST: {cls.service.HOST}")
        print(f"  PORT: {cls.service.PORT}")
        print(f"  WORKERS: {cls.service.WORKERS}")
        print(f"  DEBUG: {cls.service.DEBUG}")
        
        print("\n[LLM]")
        print(f"  API_KEY: {mask_secret(cls.llm.API_KEY)}")
        print(f"  BASE_URL: {cls.llm.BASE_URL}")
        print(f"  MODEL: {cls.llm.MODEL}")
        print(f"  TEMPERATURE: {cls.llm.TEMPERATURE}")
        print(f"  MAX_TOKENS: {cls.llm.MAX_TOKENS}")
        
        print("\n[Elasticsearch]")
        print(f"  ADDRESS: {cls.es.ADDRESS}")
        print(f"  USERNAME: {cls.es.USERNAME}")
        print(f"  PASSWORD: {mask_secret(cls.es.PASSWORD)}")
        print(f"  MEMORY_INDEX: {cls.es.MEMORY_INDEX}")

        print("\n[Qdrant]")
        print(f"  URL: {cls.qdrant.URL}")
        print(f"  API_KEY: {mask_secret(cls.qdrant.API_KEY)}")
        print(f"  COLLECTION: {cls.qdrant.COLLECTION}")
        print(f"  VECTOR_SIZE: {cls.qdrant.VECTOR_SIZE}")
        print(f"  DISTANCE: {cls.qdrant.DISTANCE}")
        
        print("\n[Embedding]")
        print(f"  HOST: {cls.embedding.HOST}")
        print(f"  PORT: {cls.embedding.PORT}")
        print(f"  MODEL: {cls.embedding.MODEL}")
        
        print("\n[Memory]")
        print(f"  OFFLINE_HOUR_RANGE: {cls.memory.OFFLINE_HOUR_RANGE}")
        print(f"  SENSORY_BUFFER_SIZE: {cls.memory.SENSORY_BUFFER_SIZE}")
        print(f"  SHORT_TERM_MAX_SIZE: {cls.memory.SHORT_TERM_MAX_SIZE}")
        print(f"  SIMILARITY_THRESHOLD: {cls.memory.SIMILARITY_THRESHOLD}")
        
        print("=" * 50)


# 快捷访问
config = Config()


if __name__ == "__main__":
    Config.print_config()

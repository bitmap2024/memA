#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统一配置：所有运行时参数从 .env 读取，组件直接 import config 即可。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(key)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _env_int(key: str, default: int) -> int:
    value = _env(key)
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    value = _env(key)
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


@dataclass
class ServiceConfig:
    HOST: str = _env("service_host", "0.0.0.0")
    PORT: int = _env_int("service_port", 51666)
    WORKERS: int = _env_int("service_workers", 8)
    DEBUG: bool = (_env("service_debug", "false") or "false").lower() == "true"


@dataclass
class LLMConfig:
    API_KEY: str = _env("deepseek_api_key", "")
    BASE_URL: str = _env("deepseek_base_url", "https://api.deepseek.com")
    MODEL: str = _env("deepseek_model", "deepseek-chat")
    TEMPERATURE: float = _env_float("deepseek_temperature", 0.4)
    MAX_TOKENS: int = _env_int("deepseek_max_tokens", 4096)
    TOP_P: float = _env_float("deepseek_top_p", 0.95)
    TIMEOUT: int = _env_int("deepseek_timeout", 60)


@dataclass
class QdrantConfig:
    URL: str = _env("qdrant_url", "http://localhost:6333")
    API_KEY: str = _env("qdrant_api_key", "")
    COLLECTION: str = _env("qdrant_collection_name", "mema")
    DENSE_DIM: int = _env_int("qdrant_dense_dim", 1024)
    DISTANCE: str = _env("qdrant_distance", "cosine")
    TIMEOUT: int = _env_int("qdrant_timeout", 30)


@dataclass
class MysqlConfig:
    HOST: str = _env("mysql_host", "127.0.0.1")
    PORT: int = _env_int("mysql_port", 3306)
    USER: str = _env("mysql_user", "root")
    PASSWORD: str = _env("mysql_password", "")
    DATABASE: str = _env("mysql_database", "memA")
    # SQLite 兼容（本地默认走 sqlite）。如果设置 mysql_use_sqlite=true，本地直接走 SQLite。
    USE_SQLITE: bool = (_env("mysql_use_sqlite", "true") or "true").lower() == "true"
    SQLITE_PATH: str = _env(
        "mysql_sqlite_path",
        str(PROJECT_ROOT / "var" / "memA.sqlite3"),
    )


@dataclass
class OssConfig:
    BUCKET: str = _env("oss_bucket_name", "memA")
    ENDPOINT: str = _env("oss_endpoint", "https://oss-cn-hangzhou.aliyuncs.com")
    ACCESS_KEY_ID: str = _env("oss_access_key_id", "")
    ACCESS_KEY_SECRET: str = _env("oss_access_key_secret", "")
    # 本地兜底目录，连不上 OSS 时写本地磁盘
    LOCAL_FALLBACK_DIR: str = _env(
        "oss_local_fallback_dir",
        str(PROJECT_ROOT / "var" / "oss_local"),
    )
    PREFIX: str = _env("oss_prefix", "chat_bot_memory")


@dataclass
class MilvusConfig:
    URI: str = _env("milvus_uri", "http://localhost:19530")
    TOKEN: str = _env("milvus_token", "")
    DATABASE: str = _env("milvus_database", "default")
    COLLECTION: str = _env("milvus_collection_name", "memA")
    DENSE_DIM: int = _env_int("milvus_dense_dim", 1024)
    METRIC_TYPE: str = _env("milvus_metric_type", "COSINE")
    TIMEOUT: int = _env_int("milvus_timeout", 30)


@dataclass
class EmbeddingConfig:
    HOST: str = _env("embedding_host", "127.0.0.1")
    PORT: int = _env_int("embedding_port", 50051)
    MODEL: str = _env("embedding_model", "bge-m3")
    POOL_SIZE: int = _env_int("embedding_pool_size", 5)
    TIMEOUT: int = _env_int("embedding_timeout", 30)


@dataclass
class CompressorConfig:
    MODEL_PATH: str = _env(
        "compressor_model_path",
        r"D:\aiworks\premodel\llmlingua-2-bert-base-multilingual-cased-meetingbank",
    )
    RATE: float = _env_float("compressor_rate", 0.85)
    MAX_TOKENS: int = _env_int("compressor_max_tokens", 512)


@dataclass
class RetrievalConfig:
    TOPIC_TOKEN_THRESHOLD: int = _env_int("topic_token_threshold", 512)
    TOPIC_SIMILARITY_THRESHOLD: float = _env_float("topic_similarity_threshold", 0.55)
    DEFAULT_TOP_K: int = _env_int("retrieval_top_k", 10)
    DENSE_SCORE_THRESHOLD: float = _env_float("retrieval_dense_threshold", 0.4)
    MERGE_SIM_THRESHOLD: float = _env_float("merge_similarity_threshold", 0.5)
    RRF_RANK_CONSTANT: int = _env_int("rrf_rank_constant", 60)
    MMR_LAMBDA: float = _env_float("mmr_lambda", 0.5)
    TIME_DECAY_TAU_DAYS: float = _env_float("time_decay_tau_days", 30.0)


@dataclass
class RerankerConfig:
    BGE_RERANK_PATH: str = _env(
        "bge_rerank_path",
        r"D:\aiworks\premodel\bge-reranker-v2-m3",
    )
    BGE_RERANK_DEVICE: str = _env("bge_rerank_device", "cpu")


@dataclass
class ExtractorConfig:
    TYPE: str = _env("extractor_type", "multikind_llm")


class Config:
    """全局配置聚合（直接 import config 使用类级属性即可）。"""

    service: ServiceConfig = ServiceConfig()
    llm: LLMConfig = LLMConfig()
    qdrant: QdrantConfig = QdrantConfig()
    milvus: MilvusConfig = MilvusConfig()
    mysql: MysqlConfig = MysqlConfig()
    oss: OssConfig = OssConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    compressor: CompressorConfig = CompressorConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    reranker: RerankerConfig = RerankerConfig()
    extractor: ExtractorConfig = ExtractorConfig()

    @classmethod
    def reload(cls) -> None:
        load_dotenv(ENV_PATH, override=True)
        cls.service = ServiceConfig()
        cls.llm = LLMConfig()
        cls.qdrant = QdrantConfig()
        cls.milvus = MilvusConfig()
        cls.mysql = MysqlConfig()
        cls.oss = OssConfig()
        cls.embedding = EmbeddingConfig()
        cls.compressor = CompressorConfig()
        cls.retrieval = RetrievalConfig()
        cls.reranker = RerankerConfig()
        cls.extractor = ExtractorConfig()

    @classmethod
    def describe(cls) -> str:
        def _mask(value: Optional[str], keep: int = 4) -> str:
            if not value:
                return "<empty>"
            if len(value) <= keep:
                return "***"
            return value[:keep] + "***"

        return "\n".join(
            [
                "=" * 50,
                "[Service]",
                f"  HOST={cls.service.HOST} PORT={cls.service.PORT}",
                "[LLM]",
                f"  MODEL={cls.llm.MODEL} BASE_URL={cls.llm.BASE_URL} API_KEY={_mask(cls.llm.API_KEY)}",
                "[Qdrant]",
                f"  URL={cls.qdrant.URL} COLLECTION={cls.qdrant.COLLECTION} DIM={cls.qdrant.DENSE_DIM}",
                "[Milvus]",
                f"  URI={cls.milvus.URI} COLLECTION={cls.milvus.COLLECTION} DIM={cls.milvus.DENSE_DIM}",
                "[MySQL]",
                f"  USE_SQLITE={cls.mysql.USE_SQLITE} HOST={cls.mysql.HOST}:{cls.mysql.PORT} DB={cls.mysql.DATABASE}",
                f"  SQLITE_PATH={cls.mysql.SQLITE_PATH}",
                "[Embedding]",
                f"  HOST={cls.embedding.HOST}:{cls.embedding.PORT} MODEL={cls.embedding.MODEL}",
                "[Compressor]",
                f"  MODEL_PATH={cls.compressor.MODEL_PATH}",
                "[OSS]",
                f"  BUCKET={cls.oss.BUCKET} ENDPOINT={cls.oss.ENDPOINT} PREFIX={cls.oss.PREFIX}",
                "=" * 50,
            ]
        )


config = Config


if __name__ == "__main__":
    print(Config.describe())

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MemoryService：统一的 facade，组合 extract / update / retrieval 三个能力。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from config.setting import Config
from src.db import MemoryStoreClient, create_memory_store
from src.db.qdrant import QdrantMemoryStore
from src.extract_mem.memory_extract_pipeline import MemoryExtractPipeline
from src.extract_mem.text_compressor import TextCompressor
from src.extract_mem.topic_segment import TopicSegmenter
from src.retrieval_mem.hybrid_retrieval import HybridRetrieval
from src.services.extract_memory_service import ExtractMemoryService
from src.services.retrieval_memory_service import RetrievalMemoryService
from src.services.update_memory_service import UpdateMemoryService
from src.update_mem.category_doc_builder import CategoryDocBuilder
from src.update_mem.sleep_mode_update import SleepModeUpdater
from src.embeddings.local.bgem3_text_embedder import BGEM3TextEmbedder

class MemoryService:
    """共享底层组件、暴露 extract / update / retrieval 入口。"""

    def __init__(self, store: Optional[MemoryStoreClient] = None) -> None:
        # 共享底层资源（连接 / 客户端）
        self.embedder = BGEM3TextEmbedder()
        self.qdrant = QdrantMemoryStore()
        self.store: MemoryStoreClient = store or create_memory_store()

        self.compressor = TextCompressor(
            model_path=Config.compressor.MODEL_PATH,
            max_tokens=Config.compressor.MAX_TOKENS,
        )
        self.segmenter = TopicSegmenter(
            embedder=self.embedder,
            token_threshold=Config.retrieval.TOPIC_TOKEN_THRESHOLD,
            similarity_threshold=Config.retrieval.TOPIC_SIMILARITY_THRESHOLD,
        )

        self.extract_pipeline = MemoryExtractPipeline(
            embedder=self.embedder,
            embedding_db=self.qdrant,
            relational_db=self.store,
            compressor=self.compressor,
            segmenter=self.segmenter,
        )
        self.hybrid_retrieval = HybridRetrieval(
            embedder=self.embedder,
            qdrant=self.qdrant,
            store=self.store,
        )

        self.extract_service = ExtractMemoryService(pipeline=self.extract_pipeline)
        self.update_service = UpdateMemoryService(
            sleep_updater=SleepModeUpdater(
                embedder=self.embedder, qdrant=self.qdrant, store=self.store
            ),
            doc_builder=CategoryDocBuilder(store=self.store),
            store=self.store,
        )
        self.retrieval_service = RetrievalMemoryService(
            hybrid=self.hybrid_retrieval, store=self.store
        )

    # ------------------------------------------------------------------
    # extract
    # ------------------------------------------------------------------
    def extract_memory(
        self,
        user_id: str,
        conversation: List[Dict[str, str]],
        conversation_date_time: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.extract_service.extract_one(
            user_id=user_id,
            conversation=conversation,
            conversation_date_time=conversation_date_time,
            session_id=session_id,
        )

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------
    def update_user_memory(
        self,
        user_id: str,
        max_workers: int = 4,
        enable_merge: bool = True,
        enable_doc: bool = True,
    ) -> Dict[str, Any]:
        return self.update_service.update_user(
            user_id=user_id,
            max_workers=max_workers,
            enable_merge=enable_merge,
            enable_doc=enable_doc,
        )

    # ------------------------------------------------------------------
    # retrieve
    # ------------------------------------------------------------------
    def retrieve_memory(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
        memory_type: Optional[str] = None,
        memory_category: Optional[str] = None,
        use_bge_rerank: bool = True,
        use_llm_rerank: bool = False,
        use_mmr: bool = True,
    ) -> List[Dict[str, Any]]:
        return self.retrieval_service.retrieve(
            user_id=user_id,
            query=query,
            top_k=top_k,
            memory_type=memory_type,
            memory_category=memory_category,
            use_bge_rerank=use_bge_rerank,
            use_llm_rerank=use_llm_rerank,
            use_mmr=use_mmr,
        )

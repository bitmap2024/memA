#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HybridRetrieval：bge-m3 dense + sparse + BM25 → RRF 融合 → 时间衰减 / importance / confidence →
（可选）bge-rerank / LLM rerank → MMR 多样化。
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from loguru import logger
import numpy as np

from config.setting import Config
from src.db import MemoryStoreClient, create_memory_store
from src.db.qdrant import QdrantMemoryStore, SparseVector
from src.embeddings.local.bgem3_text_embedder import BGEM3TextEmbedder
from src.reranker import BgeReranker, LLMReranker, RRFReranker

from .bm25_search import BM25Searcher
from .mmr_search import MMRSearch
from .time_decay import time_decay_factor


class HybridRetrieval:
    """三路召回 + RRF + 时间衰减 + 重要度/置信度加权 + rerank + MMR。"""

    def __init__(
        self,
        embedder: Optional[BGEM3TextEmbedder] = None,
        qdrant: Optional[QdrantMemoryStore] = None,
        store: Optional[MemoryStoreClient] = None,
        bm25: Optional[BM25Searcher] = None,
        bge_reranker: Optional[BgeReranker] = None,
        llm_reranker: Optional[LLMReranker] = None,
    ) -> None:
        self.embedder = embedder or BGEM3TextEmbedder()
        self.qdrant = qdrant or QdrantMemoryStore()
        self.store = store or create_memory_store()
        self.bm25 = bm25 or BM25Searcher()
        self.bge_reranker = bge_reranker  # 懒加载
        self.llm_reranker = llm_reranker  # 懒加载

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def retrieve(
        self,
        user_id: str,
        query: str,
        top_k: int = None,
        recall_size: int = None,
        memory_type: Optional[str] = None,
        memory_category: Optional[str] = None,
        use_bge_rerank: bool = True,
        use_llm_rerank: bool = False,
        use_mmr: bool = True,
        time_decay_tau_days: Optional[float] = None,
        update_retrieval_stats: bool = True,
    ) -> List[Dict[str, Any]]:
        top_k = top_k or Config.retrieval.DEFAULT_TOP_K
        recall_size = recall_size or max(top_k * 5, 30)
        tau = time_decay_tau_days if time_decay_tau_days is not None else Config.retrieval.TIME_DECAY_TAU_DAYS

        if not query or not user_id:
            return []

        # ---------- 召回 ----------
        query_dense, query_sparse = self._encode_query(query)
        payload_filter = QdrantMemoryStore.build_filter(
            user_id=user_id,
            memory_type=memory_type,
            memory_category=memory_category,
            status="active",
        )

        dense_hits = self.qdrant.search_dense(
            query_dense=query_dense,
            limit=recall_size,
            payload_filter=payload_filter,
        )
        sparse_hits = self.qdrant.search_sparse(
            sparse=query_sparse,
            limit=recall_size,
            payload_filter=payload_filter,
        )
        bm25_hits = self._bm25_search(
            user_id=user_id,
            query=query,
            recall_size=recall_size,
            memory_type=memory_type,
            memory_category=memory_category,
        )

        # ---------- RRF 融合 ----------
        rrf = RRFReranker(
            rank_constant=Config.retrieval.RRF_RANK_CONSTANT,
            id_keys=("memory_id", "_id", "id"),
        )
        normalized_dense = self._normalize(dense_hits, source="dense")
        normalized_sparse = self._normalize(sparse_hits, source="sparse")
        normalized_bm25 = self._normalize(bm25_hits, source="bm25")
        fused = rrf.rerank(
            ranked_lists=[normalized_dense, normalized_sparse, normalized_bm25],
            top_k=recall_size,
            weights=[1.0, 0.8, 0.7],
        )
        if not fused:
            return []

        # ---------- 时间衰减 + importance/confidence 加权 ----------
        now = datetime.now(timezone.utc)
        for item in fused:
            decay = time_decay_factor(
                item.get("updated_at") or item.get("created_at"), now=now, tau_days=tau
            )
            importance = float(item.get("importance") or 0.0)
            confidence = float(item.get("confidence") or 0.0)
            base = float(item.get("rrf_score") or 0.0)
            final_score = base * decay * (0.6 + 0.4 * importance) * (0.5 + 0.5 * confidence)
            item["time_decay"] = decay
            item["final_score"] = final_score
        fused.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)

        rerank_pool = fused[: max(top_k * 3, top_k)]

        # ---------- bge-rerank ----------
        if use_bge_rerank:
            reranker = self.bge_reranker or BgeReranker()
            self.bge_reranker = reranker
            rerank_pool = reranker.rerank(query=query, candidates=rerank_pool, top_k=len(rerank_pool))

        # ---------- LLM rerank ----------
        if use_llm_rerank:
            llm_rr = self.llm_reranker or LLMReranker()
            self.llm_reranker = llm_rr
            rerank_pool = llm_rr.rerank(query=query, memory_items=rerank_pool, top_k=len(rerank_pool))

        # ---------- MMR ----------
        if use_mmr and len(rerank_pool) > top_k:
            rerank_pool = self._apply_mmr(rerank_pool, query_dense, top_k=top_k)
        else:
            rerank_pool = rerank_pool[:top_k]

        # ---------- 命中统计 ----------
        if update_retrieval_stats and rerank_pool:
            try:
                self.store.bump_retrieval([str(item.get("memory_id") or item.get("_id")) for item in rerank_pool])
            except Exception as e:
                logger.warning(f"[HybridRetrieval] bump_retrieval 失败: {e}")

        return rerank_pool

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def _encode_query(self, query: str) -> tuple:
        """用 BGEM3TextEmbedder 编码 query，返回 (dense_list, SparseVector)。"""
        enc = self.embedder.encode([query])
        dense_list = (enc.get("dense") or [[]])[0]
        if hasattr(dense_list, "tolist"):
            dense_list = dense_list.tolist()
        dense_list = list(dense_list)

        sparse_raw = (enc.get("sparse") or [None])[0]
        if isinstance(sparse_raw, dict) and sparse_raw:
            sparse_vec = SparseVector(
                indices=[int(k) for k in sparse_raw.keys()],
                values=[float(v) for v in sparse_raw.values()],
            )
        else:
            sparse_vec = SparseVector()
        return dense_list, sparse_vec

    def _bm25_search(
        self,
        user_id: str,
        query: str,
        recall_size: int,
        memory_type: Optional[str],
        memory_category: Optional[str],
    ) -> List[Dict[str, Any]]:
        if not self.bm25.has_index(user_id):
            memories = self.store.list_user_memories(user_id=user_id, status="active")
            self.bm25.build_for_user(user_id=user_id, memories=memories)

        ranked = self.bm25.search(user_id=user_id, query=query, top_k=recall_size * 2)
        if not ranked:
            return []

        ids = [r["memory_id"] for r in ranked]
        # 拉 Qdrant payload，保证后续融合字段对齐
        records = self.qdrant.get_by_ids(ids=ids, with_vectors=False)
        # qdrant point _id 是派生 UUID，按 payload 里的业务 memory_id 建索引
        record_by_id = {str(rec.get("memory_id") or rec.get("_id")): rec for rec in records}

        hits: List[Dict[str, Any]] = []
        for r in ranked:
            rec = record_by_id.get(r["memory_id"])
            if not rec:
                continue
            if memory_type and rec.get("memory_type") != memory_type:
                continue
            if memory_category and rec.get("memory_category") != memory_category:
                continue
            rec = dict(rec)
            rec["bm25_score"] = float(r["score"])
            hits.append(rec)
        return hits

    @staticmethod
    def _normalize(hits: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
        normalized = []
        for hit in hits:
            data = dict(hit)
            data.setdefault("memory_id", hit.get("memory_id") or hit.get("_id"))
            score = float(hit.get("_score") or hit.get("score") or hit.get(f"{source}_score") or 0.0)
            data[f"{source}_score"] = score
            data.setdefault("score", score)
            normalized.append(data)
        return normalized

    def _apply_mmr(
        self,
        candidates: List[Dict[str, Any]],
        query_dense: Sequence[float],
        top_k: int,
        lambda_param: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        lambda_param = lambda_param if lambda_param is not None else Config.retrieval.MMR_LAMBDA

        contents = [str(c.get("memory_content") or "") for c in candidates]
        enc = self.embedder.encode(contents, return_sparse=False)
        embeddings = [
            d.tolist() if hasattr(d, "tolist") else list(d)
            for d in (enc.get("dense") or [])
        ]

        if not embeddings or not any(any(v != 0.0 for v in vec) for vec in embeddings):
            return candidates[:top_k]

        try:
            searcher = MMRSearch(lambda_param=lambda_param)
            mmr_results = searcher.search(
                query_embedding=np.asarray(query_dense),
                candidate_embeddings=np.asarray(embeddings),
                candidate_items=candidates,
                top_k=top_k,
            )
        except Exception as e:
            logger.warning(f"[HybridRetrieval] MMR 失败，回退原始顺序: {e}")
            return candidates[:top_k]

        out: List[Dict[str, Any]] = []
        for r in mmr_results:
            item = r.get("item")
            if not item:
                continue
            data = dict(item)
            data["mmr_score"] = float(r.get("mmr_score") or 0.0)
            data["mmr_relevance"] = float(r.get("relevance_score") or 0.0)
            out.append(data)
        return out

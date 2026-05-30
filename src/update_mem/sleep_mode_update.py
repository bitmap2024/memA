#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sleep Mode Updater：按 user_id 读取所有活跃记忆，构建相似矩阵，
按阈值聚类，对每个聚类调用 LLM merge prompt 生成 canonical 记忆，
归档旧记忆并写回新记忆（仅作用于第三层 memory layer）。

合并语义全部委托给 :class:`MysqlMemoryStore.merge_memories`：
聚合 source_topic_ids / source_topic_cites、归档旧记忆、记录
derived_from_memory_ids 与 derived_memory_count。Sleep mode 只负责
聚类决策、调用 LLM 以及把结果同步到向量库（Qdrant）。
"""

from __future__ import annotations
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import concurrent.futures
import json
import re
import threading
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from loguru import logger

from config.releatiion_schema import MemoryItem
from config.setting import Config
from prompt.memory_prompt.chat_bot.prompt_zh import MERGE_MEMORY_PROMPT
from src.db.mysql import MysqlMemoryStore
from src.db.qdrant import QdrantMemoryStore, SparseVector
from src.embeddings.local.bgem3_text_embedder import BGEM3TextEmbedder
from src.llm.openai_llm import LLMApi


class _UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent: Dict[str, str] = {it: it for it in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def groups(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = defaultdict(list)
        for item in self.parent:
            out[self.find(item)].append(item)
        return out


class SleepModeUpdater:
    """User 级别的记忆合并（memory layer 层内合并）。"""

    def __init__(
        self,
        embedder: Optional[BGEM3TextEmbedder],
        embedding_db: Optional[QdrantMemoryStore],
        relational_db: Optional[MysqlMemoryStore],
        llm: Optional[LLMApi],
        similarity_threshold: Optional[float] = None,
    ):
        self.embedder = embedder
        self.embedding_db = embedding_db  
        self.relational_db = relational_db
        self.llm = llm
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else Config.retrieval.MERGE_SIM_THRESHOLD
        )

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------
    def run(self, user_id: str, max_workers: int = 4) -> Dict[str, Any]:
        logger.info(f"[SleepModeUpdater] 启动 sleep mode user_id={user_id}")
        memories = self.relational_db.list_user_memories(user_id=user_id, status="active")
        if not memories:
            logger.info(f"[SleepModeUpdater] 用户无活跃记忆: {user_id}")
            return {"user_id": user_id, "clusters": 0, "merged": 0, "archived": 0}

        clusters = self._build_clusters(memories)
        logger.info(f"[SleepModeUpdater] 聚类完成 clusters={len(clusters)}")

        stats = {
            "user_id": user_id,
            "clusters": len(clusters),
            "merged": 0,
            "archived": 0,
        }
        if not clusters:
            return stats

        lock = threading.Lock()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(self._merge_cluster, user_id, cluster)
                for cluster in clusters
            ]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    merged_cnt, archived_cnt = fut.result()
                except Exception as e:
                    logger.warning(f"[SleepModeUpdater] 聚类合并异常: {e}")
                    continue
                with lock:
                    stats["merged"] += merged_cnt
                    stats["archived"] += archived_cnt

        logger.info(f"[SleepModeUpdater] 完成: {stats}")
        return stats

    # ------------------------------------------------------------------
    # 聚类
    # ------------------------------------------------------------------
    def _build_clusters(self, memories: List[MemoryItem]) -> List[List[MemoryItem]]:
        """按 (memory_type, memory_category) 分桶后基于 dense 向量做相似聚类。"""
        bucket: Dict[Tuple[str, str], List[MemoryItem]] = defaultdict(list)
        for mem in memories:
            bucket[(mem.memory_type, mem.memory_category)].append(mem)

        all_clusters: List[List[MemoryItem]] = []
        for group in bucket.values():
            if len(group) < 2:
                continue

            vectors = self._fetch_dense_vectors([m.memory_id for m in group])
            missing_ids = [m.memory_id for m in group if m.memory_id not in vectors]
            if missing_ids:
                contents_by_id = {m.memory_id: m.memory_content for m in group}
                fresh = self.embedder.encode([contents_by_id[mid] for mid in missing_ids])
                for mid, dense in zip(missing_ids, fresh.get("dense") or []):
                    vectors[mid] = dense.tolist() if hasattr(dense, "tolist") else list(dense)

            uf = _UnionFind([m.memory_id for m in group])
            for id_a, id_b in self._similar_pairs(
                [m.memory_id for m in group], vectors
            ):
                uf.union(id_a, id_b)

            id_to_mem = {m.memory_id: m for m in group}
            for ids in uf.groups().values():
                if len(ids) < 2:
                    continue
                all_clusters.append([id_to_mem[i] for i in ids])
        return all_clusters

    def _similar_pairs(
        self, ids: List[str], vectors: Dict[str, List[float]]
    ) -> List[Tuple[str, str]]:
        """用一次矩阵乘法算出整组的余弦相似度，返回超过阈值的 id 对。

        相比逐对的纯 Python 计算，这里把所有向量堆成矩阵，L2 归一化后
        ``N @ N.T`` 一次性得到相似度矩阵，再取上三角中达标的下标对。
        """
        valid_ids = [i for i in ids if vectors.get(i)]
        if len(valid_ids) < 2:
            return []

        mat = np.asarray([vectors[i] for i in valid_ids], dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        normed = mat / norms

        sim = normed @ normed.T
        mask = np.triu(sim >= self.similarity_threshold, k=1)
        rows, cols = np.nonzero(mask)
        return [(valid_ids[r], valid_ids[c]) for r, c in zip(rows, cols)]

    def _fetch_dense_vectors(self, memory_ids: List[str]) -> Dict[str, List[float]]:
        if not memory_ids:
            return {}
        try:
            records = self.embedding_db.get_by_ids(ids=memory_ids, with_vectors=True)
        except Exception as e:
            logger.warning(f"[SleepModeUpdater] 拉取向量失败，将临时重算: {e}")
            return {}
        out: Dict[str, List[float]] = {}
        for rec in records:
            # 业务 memory_id 存于 payload，point _id 是派生 UUID，不能直接用
            mem_id = str(rec.get("memory_id") or "")
            if not mem_id:
                continue
            vectors = rec.get("_vectors") or {}
            dense = vectors.get(QdrantMemoryStore.DENSE_NAME) or []
            if dense:
                out[mem_id] = list(dense)
        return out

    # ------------------------------------------------------------------
    # 合并一个聚类
    # ------------------------------------------------------------------
    def _merge_cluster(self, user_id: str, cluster: List[MemoryItem]) -> Tuple[int, int]:
        if len(cluster) < 2:
            return 0, 0

        cluster = sorted(cluster, key=lambda m: m.created_at or "", reverse=True)
        merge_result = self._call_merge_llm(cluster)
        if not merge_result.get("should_merge"):
            return 0, 0

        merged_content = (merge_result.get("merged_content") or "").strip()
        if not merged_content:
            return 0, 0

        memory_type = cluster[0].memory_type
        memory_category = cluster[0].memory_category
        importance = float(
            merge_result.get("importance") or max(m.importance for m in cluster)
        )
        confidence = float(
            merge_result.get("confidence")
            or sum(m.confidence for m in cluster) / len(cluster)
        )
        importance = max(0.0, min(1.0, importance))
        confidence = max(0.0, min(1.0, confidence))

        source_ids = [m.memory_id for m in cluster]

        # 关系库：聚合来源、归档旧记忆、写入新记忆（统一在 store 中完成）
        merged_item = self.relational_db.merge_memories(
            user_id=user_id,
            source_memory_ids=source_ids,
            merged_content=merged_content,
            memory_type=memory_type,
            memory_category=memory_category,
            importance=importance,
            confidence=confidence,
            metadata={
                "merge_reason": merge_result.get("archived_reason", ""),
                "merged_count": len(cluster),
            },
        )

        # 向量库：写入新记忆向量，并把旧记忆置为 archived
        self._sync_qdrant(merged_item, source_ids)

        return 1, len(source_ids)

    def _sync_qdrant(self, merged_item: MemoryItem, archived_ids: List[str]) -> None:
        try:
            embed = self.embedder.encode([merged_item.memory_content])
            dense_list = (embed.get("dense") or [[]])[0]
            if hasattr(dense_list, "tolist"):
                dense_list = dense_list.tolist()
            sparse_raw = (embed.get("sparse") or [None])[0]
            sparse_vec = None
            if isinstance(sparse_raw, dict) and sparse_raw:
                sparse_vec = SparseVector(
                    indices=[int(k) for k in sparse_raw.keys()],
                    values=[float(v) for v in sparse_raw.values()],
                )
            payload = {
                "memory_id": merged_item.memory_id,
                "user_id": merged_item.user_id,
                "memory_content": merged_item.memory_content,
                "memory_type": merged_item.memory_type,
                "memory_category": merged_item.memory_category,
                "status": "active",
                "importance": merged_item.importance,
                "confidence": merged_item.confidence,
                "created_at": merged_item.created_at,
                "source_topic_ids": merged_item.source_topic_ids,
            }
            self.embedding_db.upsert_one(
                point_id=merged_item.memory_id,
                dense_vector=list(dense_list),
                sparse_vector=sparse_vec,
                payload=payload,
            )
        except Exception as e:
            logger.warning(f"[SleepModeUpdater] 新记忆向量写入失败: {e}")

        for old_id in archived_ids:
            try:
                self.qdrant.set_payload(point_id=old_id, payload={"status": "archived"})
            except Exception:
                continue

    # ------------------------------------------------------------------
    # LLM 合并调用
    # ------------------------------------------------------------------
    def _call_merge_llm(self, cluster: List[MemoryItem]) -> Dict[str, Any]:
        payload = [
            {
                "id": m.memory_id,
                "memory_content": m.memory_content,
                "importance": m.importance,
                "confidence": m.confidence,
                "created_at": m.created_at,
            }
            for m in cluster
        ]
        prompt = MERGE_MEMORY_PROMPT.format(
            memory_type=cluster[0].memory_type,
            memory_category=cluster[0].memory_category,
            memories_json=json.dumps(payload, ensure_ascii=False, indent=2),
        )
        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.warning(f"[SleepModeUpdater] merge LLM 调用失败: {e}")
            return {"should_merge": False, "reason": str(e)}

        return self._parse_json(response)

    @staticmethod
    def _parse_json(response: str) -> Dict[str, Any]:
        if not response:
            return {"should_merge": False}
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", response, flags=re.DOTALL)
            if not match:
                return {"should_merge": False}
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return {"should_merge": False}

if __name__ == "__main__":
    embedder = BGEM3TextEmbedder(
        model_path="D:/aiworks/premodel/bge-m3",
        device="cuda:0",
        pooling_method="cls",
        use_fp16=True,
        max_length=8192,
        batch_size=32,
    )
    embedding_db = QdrantMemoryStore()
    relational_db = MysqlMemoryStore()
    llm = LLMApi.from_params(
        api_key=Config.llm.API_KEY,
        base_url=Config.llm.BASE_URL,
        model=Config.llm.MODEL,
        temperature=0.2,
        max_tokens=Config.llm.MAX_TOKENS,
        top_p=Config.llm.TOP_P,
        timeout=Config.llm.TIMEOUT,
    )
    updater = SleepModeUpdater(embedder, embedding_db, relational_db, llm)
    result = updater.run(user_id="09f650a56149")
    print(result)
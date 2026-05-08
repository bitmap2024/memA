#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Memory Retrieval - 在线记忆检索模块

基于 embedding 向量检索相关记忆，支持 MMR 多样性排序。
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

from loguru import logger
from zoneinfo import ZoneInfo

from api.emb_api import EmbeddingClientPool
from api.qdrant_api import QdrantMemoryClient
from retrieval_mem.mmr_search import MMRSearch


class MemoryRetrieval:
    """
    记忆检索器
    
    支持:
    1. 基于 embedding 的语义检索
    2. MMR 多样性排序
    3. 基于时间、类型的过滤
    """
    
    def __init__(
        self,
        config=None,
        collection_name: str = "light_memory_rag",
        embedding_host: str = "localhost",
        embedding_port: int = 80,
        pool_size: int = 5
    ):
        """
        初始化检索器
        
        Args:
            config: 配置对象（可选），如果提供则从 config 读取配置
            collection_name: Qdrant collection 名称
            embedding_host: Embedding 服务地址
            embedding_port: Embedding 服务端口
            pool_size: 连接池大小
        """
        if config is not None:
            collection_name = getattr(config.qdrant, "COLLECTION", collection_name)
            embedding_host = getattr(config.embedding, 'HOST', embedding_host)
            embedding_port = getattr(config.embedding, 'PORT', embedding_port)
        
        self.embedding_pool = EmbeddingClientPool(
            host=embedding_host,
            port=embedding_port,
            pool_size=pool_size
        )
        self.embedding_client = self.embedding_pool.get_client()
        self.vector_store = QdrantMemoryClient(collection_name=collection_name)
        self.mmr_search = MMRSearch(lambda_param=0.5)
    
    def retrieve(
        self,
        query: str,
        child_id: str = None,
        agent_id: str = None,
        top_k: int = 10,
        score_threshold: float = 0.78,
        use_mmr: bool = True,
        mmr_lambda: float = 0.5
    ) -> List[Dict]:
        """
        检索相关记忆
        
        Args:
            query: 查询文本
            child_id: 用户 ID 过滤
            agent_id: Agent ID 过滤
            top_k: 返回数量
            score_threshold: 分数阈值
            use_mmr: 是否使用 MMR 排序
            mmr_lambda: MMR 参数
            
        Returns:
            检索到的记忆列表
        """
        try:
            # 获取 query embedding
            query_emb = self.embedding_client.get_embeddings([query])[0].tolist()

            payload_filter = QdrantMemoryClient.build_filter(
                child_id=child_id,
                agent_id=agent_id,
            )

            hits = self.vector_store.search(
                query_vector=query_emb,
                limit=top_k * 10 if use_mmr else top_k,
                payload_filter=payload_filter,
                with_vectors=True,
            )

            logger.info(
                f"[retrieve] Qdrant 搜索参数: collection={self.vector_store.collection_name}, "
                f"child_id={child_id}, agent_id={agent_id}, top_k={top_k}, hits={len(hits)}"
            )

            for i, hit in enumerate(hits[:3]):
                score = hit.get('_score', 0.0)
                content = hit.get('memory_content', '')[:50]
                logger.info(f"[retrieve] hit[{i}]: score={score:.4f}, content={content}...")

            results = []
            filtered_count = 0
            for hit in hits:
                score = hit.get('_score', 0.0)
                if score < score_threshold:
                    filtered_count += 1
                    continue
                results.append({
                    "memory_id": hit.get("memory_id", hit.get("_id")),
                    "score": score,
                    "memory_content": hit.get("memory_content", ""),
                    "memory_type": hit.get("memory_type", ""),
                    "updated_at": hit.get("updated_at", ""),
                    "embedding": hit.get("vector", []),
                })

            logger.info(f"[retrieve] 阈值过滤: threshold={score_threshold}, 过滤掉={filtered_count}条, 剩余={len(results)}条")

            if use_mmr and len(results) > top_k:
                results = self._apply_mmr(query_emb, results, top_k, mmr_lambda)
                logger.info(f"[retrieve] MMR 排序后: {len(results)}条")
            else:
                results = results[:top_k]
            
            return results
            
        except Exception as e:
            logger.error(f"记忆检索失败: {e}")
            return []
    
    def _apply_mmr(
        self,
        query_emb: List[float],
        candidates: List[Dict],
        top_k: int,
        lambda_param: float = 0.5
    ) -> List[Dict]:
        """
        应用 MMR 算法进行多样性排序
        
        Args:
            query_emb: 查询 embedding
            candidates: 候选结果
            top_k: 返回数量
            lambda_param: MMR 参数
            
        Returns:
            MMR 排序后的结果
        """
        
        if len(candidates) <= top_k:
            return candidates

        candidate_embeddings = [c.get("embedding", []) for c in candidates]
        if not candidate_embeddings or not any(candidate_embeddings):
            return candidates[:top_k]

        try:
            mmr_results = self.mmr_search.search(
                query_embedding=query_emb,
                candidate_embeddings=candidate_embeddings,
                candidate_items=candidates,
                top_k=top_k
            )
            return [r["item"] for r in mmr_results if "item" in r]
        except Exception as e:
            logger.warning(f"MMR 排序失败，返回原始结果: {e}")
            return candidates[:top_k]
    
    def retrieve_by_type(
        self,
        child_id: str,
        memory_type: str,
        agent_id: str = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        按类型检索记忆
        
        Args:
            child_id: 用户 ID
            memory_type: 记忆类型
            agent_id: Agent ID
            limit: 返回数量限制
            
        Returns:
            记忆列表
        """
        try:
            records = self.vector_store.scroll_all(
                payload_filter=QdrantMemoryClient.build_filter(
                    child_id=child_id,
                    agent_id=agent_id,
                    memory_type=memory_type,
                ),
                limit=max(limit, 64),
                with_vectors=False,
            )
            records = QdrantMemoryClient.sort_records(records, "updated_at", reverse=True)
            return records[:limit]
        except Exception as e:
            logger.error(f"按类型检索记忆失败: {e}")
            return []
    
    def retrieve_recent(
        self,
        child_id: str,
        agent_id: str = None,
        days: int = 7,
        limit: int = 20
    ) -> List[Dict]:
        """
        检索最近的记忆
        
        Args:
            child_id: 用户 ID
            agent_id: Agent ID
            days: 天数
            limit: 返回数量限制
            
        Returns:
            记忆列表
        """
        current_time = datetime.now(ZoneInfo('UTC'))
        days_ago = (current_time - timedelta(days=days)).isoformat()

        try:
            records = self.vector_store.scroll_all(
                payload_filter=QdrantMemoryClient.build_filter(
                    child_id=child_id,
                    agent_id=agent_id,
                    updated_at_gte=days_ago,
                ),
                limit=max(limit, 64),
                with_vectors=False,
            )
            records = QdrantMemoryClient.sort_records(records, "updated_at", reverse=True)
            return records[:limit]
        except Exception as e:
            logger.error(f"检索最近记忆失败: {e}")
            return []


# 便捷函数
def create_retriever(
    collection_name: str = "light_memory_rag",
    embedding_host: str = "localhost",
    embedding_port: int = 80
) -> MemoryRetrieval:
    """创建记忆检索器实例"""
    return MemoryRetrieval(
        collection_name=collection_name,
        embedding_host=embedding_host,
        embedding_port=embedding_port
    )


if __name__ == "__main__":
    # 示例使用
    retriever = MemoryRetrieval()
    
    # 示例检索
    # results = retriever.retrieve(
    #     query="用户喜欢什么颜色",
    #     child_id="test_child",
    #     top_k=5
    # )
    # print(results)
    
    print("MemoryRetrieval 模块加载成功")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MMR (Maximal Marginal Relevance) 最大边际检索实现
MMR 是一种在检索结果中平衡相关性和多样性的算法。
公式: MMR = argmax[λ * Sim(d, q) - (1-λ) * max(Sim(d, d_i))]
其中:
    - d: 候选文档
    - q: 查询
    - d_i: 已选择的文档
    - λ: 权重参数，控制相关性和多样性的平衡
        - λ=1: 完全基于相关性
        - λ=0: 完全基于多样性
        - 通常使用 0.5-0.7 之间的值
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    计算两个向量之间的余弦相似度
    
    Args:
        vec1: 第一个向量
        vec2: 第二个向量
        
    Returns:
        余弦相似度值，范围 [-1, 1]
    """
    if vec1.ndim == 1:
        vec1 = vec1.reshape(1, -1)
    if vec2.ndim == 1:
        vec2 = vec2.reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    计算嵌入向量之间的余弦相似度矩阵
    
    Args:
        embeddings: 嵌入向量矩阵，shape (n, dim)
        
    Returns:
        相似度矩阵，shape (n, n)
    """
    return cosine_similarity(embeddings)


class MMRSearch:
    """
    MMR（最大边际检索）搜索类
    
    用于在检索结果中平衡相关性和多样性，避免返回过于相似的结果。
    
    Example:
        >>> mmr = MMRSearch(lambda_param=0.7)
        >>> results = mmr.search(
        ...     query_embedding=query_emb,
        ...     candidate_embeddings=doc_embs,
        ...     candidate_items=documents,
        ...     top_k=5
        ... )
    """
    
    def __init__(self, lambda_param: float = 0.7):
        """
        初始化 MMR 搜索器
        
        Args:
            lambda_param: 权重参数，控制相关性和多样性的平衡
                - 接近 1: 更关注与查询的相关性
                - 接近 0: 更关注结果的多样性
                - 推荐值: 0.5-0.7
        """
        if not 0 <= lambda_param <= 1:
            raise ValueError("lambda_param 必须在 [0, 1] 范围内")
        self.lambda_param = lambda_param
    
    def search(
        self,
        query_embedding: Union[np.ndarray, List[float]],
        candidate_embeddings: Union[np.ndarray, List[List[float]]],
        candidate_items: Optional[List[Any]] = None,
        top_k: int = 10,
        min_relevance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        执行 MMR 检索
        
        Args:
            query_embedding: 查询向量
            candidate_embeddings: 候选文档向量列表
            candidate_items: 候选文档列表（可选），如果不提供则返回索引
            top_k: 返回结果数量
            min_relevance: 最小相关性阈值，低于此阈值的候选将被过滤
            
        Returns:
            排序后的结果列表，每个元素包含:
                - index: 原始索引
                - item: 候选项（如果提供了 candidate_items）
                - relevance_score: 与查询的相似度
                - mmr_score: MMR 分数
        """
        # 转换为 numpy 数组
        query_embedding = np.array(query_embedding)
        candidate_embeddings = np.array(candidate_embeddings)
        
        if len(candidate_embeddings) == 0:
            return []
        
        # 确保维度正确
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        n_candidates = len(candidate_embeddings)
        top_k = min(top_k, n_candidates)
        
        # 计算所有候选与查询的相似度
        query_similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
        # query_similarities = np.array(candidate_items
        # 计算候选之间的相似度矩阵
        candidate_similarity_matrix = cosine_similarity(candidate_embeddings)
        
        # 初始化
        selected_indices: List[int] = []
        remaining_indices: List[int] = list(range(n_candidates))
        results: List[Dict[str, Any]] = []
        
        # 过滤低于最小相关性阈值的候选
        remaining_indices = [
            i for i in remaining_indices 
            if query_similarities[i] >= min_relevance
        ]
        
        if len(remaining_indices) == 0:
            logger.warning(f"所有候选的相关性都低于阈值 {min_relevance}")
            return []
        
        # 迭代选择 top_k 个结果
        while len(selected_indices) < top_k and len(remaining_indices) > 0:
            mmr_scores = []
            
            for idx in remaining_indices:
                # 计算与查询的相关性
                relevance = query_similarities[idx]
                
                # 计算与已选择文档的最大相似度（多样性惩罚）
                if len(selected_indices) == 0:
                    diversity_penalty = 0
                else:
                    diversity_penalty = max(
                        candidate_similarity_matrix[idx][s] 
                        for s in selected_indices
                    )
                
                # 计算 MMR 分数
                mmr_score = (
                    self.lambda_param * relevance 
                    - (1 - self.lambda_param) * diversity_penalty
                )
                mmr_scores.append((idx, mmr_score, relevance))
            
            # 选择 MMR 分数最高的候选
            best_idx, best_mmr_score, best_relevance = max(
                mmr_scores, key=lambda x: x[1]
            )
            
            # 添加到结果
            result = {
                "index": best_idx,
                "relevance_score": float(best_relevance),
                "mmr_score": float(best_mmr_score)
            }
            if candidate_items is not None:
                result["item"] = candidate_items[best_idx]
            
            results.append(result)
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        return results
    
    def rerank(
        self,
        query_embedding: Union[np.ndarray, List[float]],
        candidate_embeddings: Union[np.ndarray, List[List[float]]],
        candidate_items: List[Any],
        top_k: int = 10,
        initial_scores: Optional[List[float]] = None,
        score_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        使用 MMR 对已有检索结果进行重排序
        
        可以结合原始检索分数进行混合排序
        
        Args:
            query_embedding: 查询向量
            candidate_embeddings: 候选文档向量列表
            candidate_items: 候选文档列表
            top_k: 返回结果数量
            initial_scores: 原始检索分数（可选）
            score_weight: 原始分数的权重（当提供 initial_scores 时使用）
            
        Returns:
            重排序后的结果列表
        """
        # 转换为 numpy 数组
        query_embedding = np.array(query_embedding)
        candidate_embeddings = np.array(candidate_embeddings)
        
        if len(candidate_embeddings) == 0:
            return []
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        n_candidates = len(candidate_embeddings)
        top_k = min(top_k, n_candidates)
        
        # 计算语义相似度
        semantic_similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
        
        # 如果提供了原始分数，进行归一化并混合
        if initial_scores is not None:
            initial_scores = np.array(initial_scores)
            # 归一化到 [0, 1]
            if initial_scores.max() > initial_scores.min():
                normalized_scores = (initial_scores - initial_scores.min()) / (initial_scores.max() - initial_scores.min())
            else:
                normalized_scores = np.ones_like(initial_scores)
            
            # 混合分数
            combined_similarities = (
                score_weight * normalized_scores 
                + (1 - score_weight) * semantic_similarities
            )
        else:
            combined_similarities = semantic_similarities
        
        # 计算候选之间的相似度矩阵
        candidate_similarity_matrix = cosine_similarity(candidate_embeddings)
        
        # MMR 选择
        selected_indices: List[int] = []
        remaining_indices: List[int] = list(range(n_candidates))
        results: List[Dict[str, Any]] = []
        
        while len(selected_indices) < top_k and len(remaining_indices) > 0:
            mmr_scores = []
            
            for idx in remaining_indices:
                relevance = combined_similarities[idx]
                
                if len(selected_indices) == 0:
                    diversity_penalty = 0
                else:
                    diversity_penalty = max(
                        candidate_similarity_matrix[idx][s] 
                        for s in selected_indices
                    )
                
                mmr_score = (
                    self.lambda_param * relevance 
                    - (1 - self.lambda_param) * diversity_penalty
                )
                mmr_scores.append((idx, mmr_score, relevance))
            
            best_idx, best_mmr_score, best_relevance = max(
                mmr_scores, key=lambda x: x[1]
            )
            
            result = {
                "index": best_idx,
                "item": candidate_items[best_idx],
                "relevance_score": float(best_relevance),
                "semantic_score": float(semantic_similarities[best_idx]),
                "mmr_score": float(best_mmr_score)
            }
            if initial_scores is not None:
                result["initial_score"] = float(initial_scores[best_idx])
            
            results.append(result)
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        return results


def mmr_search(
    query_embedding: Union[np.ndarray, List[float]],
    candidate_embeddings: Union[np.ndarray, List[List[float]]],
    candidate_items: Optional[List[Any]] = None,
    top_k: int = 10,
    lambda_param: float = 0.7,
    min_relevance: float = 0.0
) -> List[Dict[str, Any]]:
    """
    MMR 检索的便捷函数
    
    Args:
        query_embedding: 查询向量
        candidate_embeddings: 候选文档向量列表
        candidate_items: 候选文档列表（可选）
        top_k: 返回结果数量
        lambda_param: 权重参数（0-1），越大越关注相关性
        min_relevance: 最小相关性阈值
        
    Returns:
        排序后的结果列表
        
    Example:
        >>> results = mmr_search(
        ...     query_embedding=query_emb,
        ...     candidate_embeddings=doc_embs,
        ...     candidate_items=documents,
        ...     top_k=5,
        ...     lambda_param=0.7
        ... )
    """
    searcher = MMRSearch(lambda_param=lambda_param)
    return searcher.search(
        query_embedding=query_embedding,
        candidate_embeddings=candidate_embeddings,
        candidate_items=candidate_items,
        top_k=top_k,
        min_relevance=min_relevance
    )


def mmr_select_indices(
    similarity_to_query: Union[np.ndarray, List[float]],
    similarity_matrix: np.ndarray,
    top_k: int = 10,
    lambda_param: float = 0.7
) -> List[int]:
    """
    基于预计算的相似度矩阵进行 MMR 选择，返回选中的索引
    
    当已经有预计算的相似度时使用此函数可以避免重复计算
    
    Args:
        similarity_to_query: 每个候选与查询的相似度
        similarity_matrix: 候选之间的相似度矩阵
        top_k: 返回的数量
        lambda_param: 权重参数
        
    Returns:
        选中的索引列表
    """
    similarity_to_query = np.array(similarity_to_query)
    n_candidates = len(similarity_to_query)
    top_k = min(top_k, n_candidates)
    
    selected_indices: List[int] = []
    remaining_indices: List[int] = list(range(n_candidates))
    
    while len(selected_indices) < top_k and len(remaining_indices) > 0:
        mmr_scores = []
        
        for idx in remaining_indices:
            relevance = similarity_to_query[idx]
            
            if len(selected_indices) == 0:
                diversity_penalty = 0
            else:
                diversity_penalty = max(
                    similarity_matrix[idx][s] 
                    for s in selected_indices
                )
            
            mmr_score = (
                lambda_param * relevance 
                - (1 - lambda_param) * diversity_penalty
            )
            mmr_scores.append((idx, mmr_score))
        
        best_idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
    
    return selected_indices


class MMRMemorySearch:
    """
    专门用于记忆检索的 MMR 搜索类
    
    集成了 embedding 计算，可以直接使用文本进行搜索
    
    Example:
        >>> from src.lib.client.embedding_client import EmbeddingClientPool
        >>> embedding_client = EmbeddingClientPool(host="xxx").get_client()
        >>> mmr_memory = MMRMemorySearch(embedding_client, lambda_param=0.7)
        >>> results = mmr_memory.search_memories(
        ...     query="用户喜欢什么",
        ...     memories=memory_list,
        ...     content_key="memory_content",
        ...     top_k=5
        ... )
    """
    
    def __init__(self, embedding_client, lambda_param: float = 0.7):
        """
        初始化
        
        Args:
            embedding_client: embedding 客户端，需要有 get_embeddings 方法
            lambda_param: MMR 权重参数
        """
        self.embedding_client = embedding_client
        self.lambda_param = lambda_param
        self.mmr_searcher = MMRSearch(lambda_param=lambda_param)
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """获取文本的 embedding"""
        embeddings = self.embedding_client.get_embeddings(texts)
        if hasattr(embeddings, 'tolist'):
            return np.array(embeddings.tolist())
        return np.array(embeddings)
    
    def search_memories(
        self,
        query: str,
        memories: List[Dict[str, Any]],
        content_key: str = "memory_content",
        embedding_key: Optional[str] = "emb",
        top_k: int = 10,
        min_relevance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        对记忆列表进行 MMR 检索
        
        Args:
            query: 查询文本
            memories: 记忆列表
            content_key: 记忆内容字段名
            embedding_key: embedding 字段名（如果记忆中已有 embedding）
            top_k: 返回数量
            min_relevance: 最小相关性阈值
            
        Returns:
            MMR 排序后的记忆列表，每个元素增加了 relevance_score 和 mmr_score
        """
        if len(memories) == 0:
            return []
        
        # 获取查询 embedding
        query_embedding = self.get_embeddings([query])[0]
        
        # 获取记忆 embeddings
        if embedding_key and all(embedding_key in m and m[embedding_key] is not None for m in memories):
            # 使用已有的 embeddings
            candidate_embeddings = np.array([m[embedding_key] for m in memories])
        else:
            # 计算 embeddings
            texts = [m[content_key] for m in memories]
            candidate_embeddings = self.get_embeddings(texts)
        
        # 执行 MMR 搜索
        results = self.mmr_searcher.search(
            query_embedding=query_embedding,
            candidate_embeddings=candidate_embeddings,
            candidate_items=memories,
            top_k=top_k,
            min_relevance=min_relevance
        )
        
        # 格式化返回结果
        formatted_results = []
        for r in results:
            memory = r["item"].copy()
            memory["relevance_score"] = r["relevance_score"]
            memory["mmr_score"] = r["mmr_score"]
            formatted_results.append(memory)
        
        return formatted_results
    
    def search_with_diversity(
        self,
        query: str,
        memories: List[Dict[str, Any]],
        content_key: str = "memory_content",
        top_k: int = 10,
        diversity_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        带多样性阈值的搜索
        
        当结果与已选结果的相似度超过阈值时停止选择
        
        Args:
            query: 查询文本
            memories: 记忆列表
            content_key: 记忆内容字段名
            top_k: 最大返回数量
            diversity_threshold: 多样性阈值，超过此相似度的结果将被跳过
            
        Returns:
            多样化的记忆列表
        """
        if len(memories) == 0:
            return []
        
        # 获取 embeddings
        query_embedding = self.get_embeddings([query])[0]
        texts = [m[content_key] for m in memories]
        candidate_embeddings = self.get_embeddings(texts)
        
        # 计算相似度
        query_embedding_2d = query_embedding.reshape(1, -1)
        query_similarities = cosine_similarity(query_embedding_2d, candidate_embeddings)[0]
        candidate_similarity_matrix = cosine_similarity(candidate_embeddings)
        
        # 带多样性阈值的选择
        selected_indices: List[int] = []
        remaining_indices: List[int] = list(range(len(memories)))
        
        # 按相关性排序
        sorted_by_relevance = sorted(
            remaining_indices, 
            key=lambda i: query_similarities[i], 
            reverse=True
        )
        
        for idx in sorted_by_relevance:
            if len(selected_indices) >= top_k:
                break
            
            # 检查与已选结果的相似度
            if len(selected_indices) == 0:
                selected_indices.append(idx)
            else:
                max_sim_to_selected = max(
                    candidate_similarity_matrix[idx][s] 
                    for s in selected_indices
                )
                if max_sim_to_selected < diversity_threshold:
                    selected_indices.append(idx)
        
        # 格式化返回
        results = []
        for idx in selected_indices:
            memory = memories[idx].copy()
            memory["relevance_score"] = float(query_similarities[idx])
            results.append(memory)
        
        return results


# # 使用示例
# if __name__ == "__main__":
#     # 示例 1: 基本 MMR 搜索
#     print("=" * 50)
#     print("示例 1: 基本 MMR 搜索")
#     print("=" * 50)
    
#     # 模拟数据
#     np.random.seed(42)
#     query_emb = np.random.rand(768)
#     doc_embs = np.random.rand(10, 768)
#     documents = [f"文档_{i}" for i in range(10)]
    
#     # 执行 MMR 搜索
#     results = mmr_search(
#         query_embedding=query_emb,
#         candidate_embeddings=doc_embs,
#         candidate_items=documents,
#         top_k=5,
#         lambda_param=0.7
#     )
    
#     print("MMR 搜索结果:")
#     for i, r in enumerate(results):
#         print(f"  {i+1}. {r['item']} (相关性: {r['relevance_score']:.4f}, MMR: {r['mmr_score']:.4f})")
    
#     # 示例 2: 使用类接口
#     print("\n" + "=" * 50)
#     print("示例 2: 使用 MMRSearch 类")
#     print("=" * 50)
    
#     mmr = MMRSearch(lambda_param=0.5)  # 更平衡的参数
#     results = mmr.search(
#         query_embedding=query_emb,
#         candidate_embeddings=doc_embs,
#         candidate_items=documents,
#         top_k=5
#     )
    
#     print("更平衡的 MMR 搜索结果 (lambda=0.5):")
#     for i, r in enumerate(results):
#         print(f"  {i+1}. {r['item']} (相关性: {r['relevance_score']:.4f}, MMR: {r['mmr_score']:.4f})")
    
#     # 示例 3: 对比不同 lambda 值
#     print("\n" + "=" * 50)
#     print("示例 3: 不同 lambda 值的对比")
#     print("=" * 50)
    
#     for lam in [0.0, 0.5, 1.0]:
#         results = mmr_search(
#             query_embedding=query_emb,
#             candidate_embeddings=doc_embs,
#             candidate_items=documents,
#             top_k=3,
#             lambda_param=lam
#         )
#         items = [r['item'] for r in results]
#         print(f"  lambda={lam}: {items}")

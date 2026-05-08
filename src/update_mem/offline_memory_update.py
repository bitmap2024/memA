import json
import concurrent.futures
import threading
from loguru import logger
from typing import List, Dict, Optional
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
from zoneinfo import ZoneInfo
from api.llm_api import LLMApi
from config.mem_prompt import UPDATE_MEMORY_PROMPT, MERGE_MEMORY_PROMPT
from config.config import Config
from api.qdrant_api import QdrantMemoryClient
from api.emb_api import EmbeddingClient


class OfflineMemoryUpdate:
    
    # 记忆类型映射
    memory_type_mapper = {
        "事实记忆": "factual",
        "事实": "factual",
        "Factual": "factual",
        "偏好记忆": "preference",
        "偏好": "preference",
        "Preference": "preference",
        "能力与发展": "ability",
        "能力": "ability",
        "发展": "ability",
        "Ability": "ability",
        "Ability & Development": "ability",
        "社会关系": "relationship",
        "关系": "relationship",
        "Relationship": "relationship",
        "性格与画像": "portrait",
        "性格": "portrait",
        "画像": "portrait",
        "Portrait": "portrait"
    }

    def __init__(self, config: Config):
        try:
            self.embedding_model = EmbeddingClient(host=config.embedding.HOST, port=config.embedding.PORT)
            self.vector_store = QdrantMemoryClient(
                collection_name=config.qdrant.COLLECTION,
                vector_size=config.qdrant.VECTOR_SIZE,
                distance=config.qdrant.DISTANCE,
            )
            self.llm = LLMApi.from_params(
                api_key=config.llm.API_KEY,
                base_url=config.llm.BASE_URL,
                model=config.llm.MODEL,
                temperature=0.7
            )
        except Exception as e:
           raise Exception(f"Error occurred during EmbeddingMergerMemory initialization: {e}")


    def get_embeddings(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        if len(texts) > 1:
            embs = self.embedding_model.get_embeddings(texts).tolist()
        elif len(texts) <= 1:
            embs = self.embedding_model.get_embeddings(texts).tolist()
        return embs

    @staticmethod
    def _to_timestamp(iso_time: str) -> float:
        if not iso_time:
            return 0.0
        try:
            return datetime.fromisoformat(iso_time.replace('Z', '+00:00')).timestamp()
        except Exception:
            return 0.0

    def get_all_memories_with_vector(
        self, 
        child_id: Optional[str] = None, 
        agent_id: Optional[str] = None,
        scroll_size: int = 1000
    ) -> List[Dict]:
        """
        获取所有记忆条目（包含向量），使用 Qdrant scroll 处理大量数据
        
        Args:
            child_id: 可选，按 child_id 过滤
            agent_id: 可选，按 agent_id 过滤
            scroll_size: 每次滚动获取的数量
            
        Returns:
            包含 id, payload(memory_content, memory_type 等), vector, float_time_stamp 的条目列表
        """
        all_entries = []
        try:
            records = self.vector_store.scroll_all(
                payload_filter=QdrantMemoryClient.build_filter(
                    child_id=child_id,
                    agent_id=agent_id,
                ),
                limit=scroll_size,
                with_vectors=True,
            )
            for record in records:
                updated_at = record.get("updated_at", "")
                float_time_stamp = self._to_timestamp(updated_at)
                all_entries.append({
                    "id": record.get("_id"),
                    "payload": {
                        "memory_id": record.get("memory_id"),
                        "child_id": record.get("child_id"),
                        "agent_id": record.get("agent_id"),
                        "session_id": record.get("session_id"),
                        "memory_content": record.get("memory_content"),
                        "memory_type": record.get("memory_type"),
                        "created_at": record.get("created_at"),
                        "updated_at": updated_at,
                        "merged": record.get("merged", False),
                        "metion_count": record.get("metion_count", 1),
                        "update_queue": record.get("update_queue", []),
                        "float_time_stamp": float_time_stamp,
                    },
                    "vector": record.get("vector", []),
                })
            
            logger.info(f"[get_all_memories_with_vector] 获取到 {len(all_entries)} 条记忆")
            return all_entries
            
        except Exception as e:
            logger.error(f"[get_all_memories_with_vector] 获取记忆失败: {e}")
            return []

    def _knn_search_with_filter(
        self, 
        query_vector: List[float], 
        limit: int, 
        max_timestamp: float,
        child_id: Optional[str] = None,
        agent_id: Optional[str] = None
    ) -> List[Dict]:
        """
        使用 KNN 搜索，带时间戳过滤（只搜索时间戳更早的记忆）
        
        Args:
            query_vector: 查询向量
            limit: 返回数量
            max_timestamp: 最大时间戳（只返回时间戳小于等于此值的记忆）
            child_id: 可选，按 child_id 过滤
            agent_id: 可选，按 agent_id 过滤
            
        Returns:
            搜索结果列表
        """
        try:
            hits = self.vector_store.search(
                query_vector=query_vector,
                limit=limit * 2,
                payload_filter=QdrantMemoryClient.build_filter(
                    child_id=child_id,
                    agent_id=agent_id,
                ),
                with_vectors=False,
            )
            results = []
            for hit in hits:
                updated_at = hit.get('updated_at', '')
                float_time_stamp = self._to_timestamp(updated_at)
                if float_time_stamp <= max_timestamp:
                    results.append({
                        "id": hit.get('_id'),
                        "score": hit.get('_score', 0.0),
                        "payload": {
                            "memory_id": hit.get("memory_id"),
                            "memory_content": hit.get("memory_content"),
                            "memory_type": hit.get("memory_type"),
                            "updated_at": updated_at,
                            "child_id": hit.get("child_id"),
                            "agent_id": hit.get("agent_id"),
                            "session_id": hit.get("session_id"),
                            "update_queue": hit.get("update_queue", []),
                        },
                        "float_time_stamp": float_time_stamp
                    })
                if len(results) >= limit:
                    break
            return results
            
        except Exception as e:
            logger.error(f"[_knn_search_with_filter] KNN 搜索失败: {e}")
            return []

    def construct_update_queue_all_entries(
        self, 
        child_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        top_k: int = 20, 
        keep_top_n: int = 10, 
        max_workers: int = 8
    ):
        """
        离线构建所有记忆条目的更新队列（并行处理）
        每个条目基于时间戳更早的记忆条目构建其 update_queue
        
        Args:
            child_id: 可选，按 child_id 过滤
            agent_id: 可选，按 agent_id 过滤
            top_k: 每个条目考虑的最近邻数量
            keep_top_n: 保留在 update_queue 中的条目数量
            max_workers: 最大线程数
        """
        call_id = f"construct_queue_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        logger.info(f"========== START {call_id} ==========")
        logger.info(f"[{call_id}] 参数: top_k={top_k}, keep_top_n={keep_top_n}, max_workers={max_workers}")
        
        # 获取所有记忆条目
        all_entries = self.get_all_memories_with_vector(child_id=child_id, agent_id=agent_id)
        logger.info(f"[{call_id}] 从数据库获取到 {len(all_entries)} 条记忆")
        
        if not all_entries:
            logger.warning(f"[{call_id}] 未找到记忆条目，跳过队列构建")
            logger.info(f"========== END {call_id} ==========")
            return
        
        updated_count = 0
        skipped_count = 0
        nonempty_queue_count = 0
        empty_queue_count = 0
        lock = threading.Lock()
        write_lock = threading.Lock()
        
        def _update_queue_construction(entry):
            nonlocal updated_count, skipped_count, nonempty_queue_count, empty_queue_count
            
            eid = entry["id"]
            payload = entry["payload"]
            vec = entry.get("vector")
            ts = payload.get("float_time_stamp", None)
            entry_child_id = payload.get("child_id")
            entry_agent_id = payload.get("agent_id")
            
            if vec is None or ts is None or ts == 0.0:
                logger.debug(f"[{call_id}] 跳过条目 {eid}: vector={vec is None}, float_time_stamp={ts}")
                with lock:
                    skipped_count += 1
                return
            
            # 搜索时间戳更早的相似记忆
            hits = self._knn_search_with_filter(
                query_vector=vec,
                limit=top_k,
                max_timestamp=ts,
                child_id=entry_child_id,
                agent_id=entry_agent_id
            )
            
            # 构建候选列表（排除自身）
            candidates = []
            for h in hits:
                hid = h["id"]
                if hid == eid:
                    continue
                candidates.append({"id": hid, "score": h.get("score", 0.0)})
            
            # 按得分排序并保留 top_n
            candidates.sort(key=lambda x: x["score"], reverse=True)
            update_queue = candidates[:keep_top_n]
            
            # 更新 payload
            new_payload = dict(payload)
            new_payload["update_queue"] = update_queue
            
            if update_queue:
                with lock:
                    nonempty_queue_count += 1
                logger.debug(f"[{call_id}] 条目 {eid} update_queue 长度={len(update_queue)}, top3={update_queue[:3]}")
            else:
                with lock:
                    empty_queue_count += 1
                logger.debug(f"[{call_id}] 条目 {eid} 过滤后无候选")
            
            # 更新 Qdrant 中的 payload
            with write_lock:
                try:
                    self.vector_store.set_payload(
                        point_id=eid,
                        payload={"update_queue": update_queue},
                    )
                except Exception as e:
                    logger.error(f"[{call_id}] 更新条目 {eid} 失败: {e}")
                    return
            
            with lock:
                updated_count += 1
        
        logger.info(f"[{call_id}] 开始并行队列构建，使用 {max_workers} 个 worker")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(_update_queue_construction, all_entries), total=len(all_entries), desc="构建更新队列"))
        
        logger.info(
            f"[{call_id}] 队列构建完成: 更新 {updated_count} 条, 跳过 {skipped_count} 条, "
            f"非空队列={nonempty_queue_count}, 空队列={empty_queue_count}"
        )
        logger.info(f"========== END {call_id} ==========")

    def _call_update_llm(self, target_memory: Dict, candidate_sources: List[Dict]) -> Optional[Dict]:
        """
        调用 LLM 进行记忆更新决策
        
        Args:
            target_memory: 目标记忆条目
            candidate_sources: 候选记忆来源列表
            
        Returns:
            LLM 返回的决策结果，包含 action 和可选的 new_memory
        """
        target_content = target_memory["payload"].get("memory_content", "")
        candidate_contents = [
            src["payload"].get("memory_content", "") 
            for src in candidate_sources
        ]
        
        prompt = UPDATE_MEMORY_PROMPT.format(
            target_memory=target_content,
            candidate_memories=json.dumps(candidate_contents, ensure_ascii=False, indent=2)
        )
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.chat(
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            # 解析 JSON 响应
            result = json.loads(response)
            
            # 获取 token 使用信息（如果可用）
            usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
            
            result["usage"] = usage
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"[_call_update_llm] JSON 解析失败: {e}, response: {response}")
            return None
        except Exception as e:
            logger.error(f"[_call_update_llm] LLM 调用失败: {e}")
            return None

    def offline_update_all_entries(
        self,
        child_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        score_threshold: float = 0.9,
        max_workers: int = 5
    ):
        """
        基于 update_queue 对所有记忆条目进行离线更新（并行处理）
        
        Args:
            child_id: 可选，按 child_id 过滤
            agent_id: 可选，按 agent_id 过滤
            score_threshold: 最小相似度阈值，用于筛选候选
            max_workers: 最大线程数
        """
        call_id = f"offline_update_all_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        logger.info(f"========== START {call_id} ==========")
        logger.info(f"[{call_id}] 参数: score_threshold={score_threshold}, max_workers={max_workers}")
        
        # 获取所有记忆条目
        all_entries = self.get_all_memories_with_vector(child_id=child_id, agent_id=agent_id)
        logger.info(f"[{call_id}] 从数据库获取到 {len(all_entries)} 条记忆")
        
        if not all_entries:
            logger.warning(f"[{call_id}] 未找到记忆条目，跳过离线更新")
            logger.info(f"========== END {call_id} ==========")
            return
        
        processed_count = 0
        updated_count = 0
        deleted_count = 0
        skipped_count = 0
        lock = threading.Lock()
        write_lock = threading.Lock()
        
        update_token_stats = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        
        token_lock = threading.Lock()
        
        def update_entry(entry):
            nonlocal processed_count, updated_count, deleted_count, skipped_count
            
            eid = entry["id"]
            payload = entry["payload"]
            
            # 反向查找：遍历所有条目的 update_queue，找出哪些条目的队列中包含当前条目
            candidate_sources = []
            for other in all_entries:
                update_queue = other["payload"].get("update_queue", [])
                for candidate in update_queue:
                    if candidate.get("id") == eid and candidate.get("score", 0) >= score_threshold:
                        candidate_sources.append(other)
                        break
            
            if not candidate_sources:
                with lock:
                    skipped_count += 1
                return
            
            with lock:
                processed_count += 1
            
            # 调用 LLM 进行更新决策
            updated_entry = self._call_update_llm(entry, candidate_sources)
            
            if updated_entry is None:
                return
            
            # 记录 token 消耗
            usage = updated_entry.get("usage", {})
            with token_lock:
                update_token_stats["calls"] += 1
                update_token_stats["prompt_tokens"] += usage.get("prompt_tokens", 0)
                update_token_stats["completion_tokens"] += usage.get("completion_tokens", 0)
                update_token_stats["total_tokens"] += usage.get("total_tokens", 0)
            
            logger.debug(f"[{call_id}] LLM 调用 {eid} - Tokens: {usage.get('total_tokens', 0)}")
            
            action = updated_entry.get("action")
            
            if action == "delete":
                # 删除记忆
                with write_lock:
                    try:
                        self.vector_store.delete_one(eid)
                        with lock:
                            deleted_count += 1
                        logger.debug(f"[{call_id}] 删除条目: {eid}")
                    except Exception as e:
                        logger.error(f"[{call_id}] 删除条目 {eid} 失败: {e}")
                        
            elif action == "update":
                # 更新记忆内容
                new_memory = updated_entry.get("new_memory")
                if new_memory:
                    with write_lock:
                        try:
                            self.vector_store.set_payload(
                                point_id=eid,
                                payload={
                                    "memory_content": new_memory,
                                    "updated_at": datetime.now(ZoneInfo('UTC')).isoformat(),
                                },
                            )
                            with lock:
                                updated_count += 1
                            logger.debug(f"[{call_id}] 更新条目: {eid}")
                        except Exception as e:
                            logger.error(f"[{call_id}] 更新条目 {eid} 失败: {e}")
            # action == "ignore" 时不做任何操作
        
        logger.info(f"[{call_id}] 开始并行离线更新，使用 {max_workers} 个 worker")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(update_entry, all_entries), total=len(all_entries), desc="离线更新记忆"))
        
        logger.info(f"[{call_id}] 离线更新完成:")
        logger.info(f"[{call_id}]   - 处理: {processed_count} 条")
        logger.info(f"[{call_id}]   - 更新: {updated_count} 条")
        logger.info(f"[{call_id}]   - 删除: {deleted_count} 条")
        logger.info(f"[{call_id}]   - 跳过（无候选）: {skipped_count} 条")
        logger.info(
            f"[{call_id}]   - LLM 调用次数: {update_token_stats['calls']}, "
            f"总 tokens: {update_token_stats['total_tokens']}"
        )
        logger.info(f"========== END {call_id} ==========")

    def _find_similar_memory_clusters(
        self,
        entries: List[Dict],
        score_threshold: float = 0.9
    ) -> List[List[Dict]]:
        """
        找出相似记忆的聚类
        
        基于 embedding 相似度，将相似的记忆聚类在一起
        
        Args:
            entries: 记忆条目列表
            score_threshold: 相似度阈值
            
        Returns:
            聚类列表，每个聚类是相似记忆的列表
        """
        if not entries:
            return []
        
        # 按 (child_id, agent_id, memory_type) 分组
        grouped = defaultdict(list)
        for entry in entries:
            payload = entry.get("payload", {})
            key = (
                payload.get("child_id"),
                payload.get("agent_id"),
                payload.get("memory_type")
            )
            grouped[key].append(entry)
        
        all_clusters = []
        
        for group_key, group_entries in grouped.items():
            if len(group_entries) < 2:
                continue
            
            # 使用并查集进行聚类
            parent = {e["id"]: e["id"] for e in group_entries}
            
            def find(x):
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]
            
            def union(x, y):
                px, py = find(x), find(y)
                if px != py:
                    parent[px] = py
            
            # 遍历每对记忆，检查相似度
            for i, entry in enumerate(group_entries):
                update_queue = entry.get("payload", {}).get("update_queue", [])
                for candidate in update_queue:
                    if candidate.get("score", 0) >= score_threshold:
                        cid = candidate.get("id")
                        if cid in parent:
                            union(entry["id"], cid)
            
            # 构建聚类
            clusters_map = defaultdict(list)
            for entry in group_entries:
                root = find(entry["id"])
                clusters_map[root].append(entry)
            
            # 只保留 size >= 2 的聚类
            for cluster in clusters_map.values():
                if len(cluster) >= 2:
                    all_clusters.append(cluster)
        
        return all_clusters

    def _call_merge_llm(self, memories: List[Dict], memory_type: str) -> Optional[Dict]:
        """
        调用 LLM 进行记忆合并
        
        Args:
            memories: 待合并的记忆列表
            memory_type: 记忆类型
            
        Returns:
            LLM 返回的合并结果
        """
        memory_contents = [
            m.get("payload", {}).get("memory_content", "") 
            for m in memories
        ]
        
        prompt = MERGE_MEMORY_PROMPT.format(
            memories_to_merge=json.dumps(memory_contents, ensure_ascii=False, indent=2),
            memory_type=memory_type
        )
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.chat(
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response)
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"[_call_merge_llm] JSON 解析失败: {e}, response: {response}")
            return None
        except Exception as e:
            logger.error(f"[_call_merge_llm] LLM 调用失败: {e}")
            return None

    def merge_similar_memories(
        self,
        child_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        score_threshold: float = 0.9,
        max_workers: int = 5
    ) -> Dict:
        """
        合并相似记忆（Sleep Mode 核心功能）
        
        将相似度高于阈值的记忆合并成一条，删除冗余记忆
        
        Args:
            child_id: 可选，按 child_id 过滤
            agent_id: 可选，按 agent_id 过滤
            score_threshold: 相似度阈值
            max_workers: 最大线程数
            
        Returns:
            合并统计信息
        """
        call_id = f"merge_memories_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        logger.info(f"========== START {call_id} ==========")
        logger.info(f"[{call_id}] 参数: score_threshold={score_threshold}, max_workers={max_workers}")
        
        # 获取所有记忆条目
        all_entries = self.get_all_memories_with_vector(child_id=child_id, agent_id=agent_id)
        logger.info(f"[{call_id}] 从数据库获取到 {len(all_entries)} 条记忆")
        
        if not all_entries:
            logger.warning(f"[{call_id}] 未找到记忆条目，跳过合并")
            logger.info(f"========== END {call_id} ==========")
            return {"clusters": 0, "merged": 0, "deleted": 0}
        
        # 找出相似记忆的聚类
        clusters = self._find_similar_memory_clusters(all_entries, score_threshold)
        logger.info(f"[{call_id}] 找到 {len(clusters)} 个待合并的记忆聚类")
        
        if not clusters:
            logger.info(f"[{call_id}] 没有需要合并的记忆聚类")
            logger.info(f"========== END {call_id} ==========")
            return {"clusters": 0, "merged": 0, "deleted": 0}
        
        merged_count = 0
        deleted_count = 0
        failed_count = 0
        lock = threading.Lock()
        write_lock = threading.Lock()
        
        def merge_cluster(cluster: List[Dict]):
            nonlocal merged_count, deleted_count, failed_count
            
            if len(cluster) < 2:
                return
            
            # 获取记忆类型（同一聚类中类型相同）
            memory_type = cluster[0].get("payload", {}).get("memory_type", "")
            
            # 按时间排序，最新的在前
            cluster.sort(
                key=lambda x: x.get("payload", {}).get("float_time_stamp", 0),
                reverse=True
            )
            
            # 调用 LLM 合并
            merge_result = self._call_merge_llm(cluster, memory_type)
            
            if merge_result is None:
                with lock:
                    failed_count += 1
                return
            
            should_merge = merge_result.get("should_merge", False)
            
            if not should_merge:
                logger.debug(f"[{call_id}] LLM 判定不需要合并")
                return
            
            merged_memory = merge_result.get("merged_memory", "")
            if not merged_memory:
                logger.warning(f"[{call_id}] 合并结果为空")
                return
            
            # 保留最新的记忆，更新其内容
            newest_entry = cluster[0]
            newest_id = newest_entry["id"]
            
            # 获取新的 embedding
            try:
                new_embedding = self.embedding_model.get_embeddings([merged_memory])[0].tolist()
            except Exception as e:
                logger.error(f"[{call_id}] 获取 embedding 失败: {e}")
                with lock:
                    failed_count += 1
                return
            
            with write_lock:
                try:
                    # 更新最新记忆的内容和 embedding
                    merged_payload = dict(newest_entry.get("payload", {}))
                    merged_payload.update(
                        {
                            "memory_content": merged_memory,
                            "updated_at": datetime.now(ZoneInfo('UTC')).isoformat(),
                            "merged": True,
                        }
                    )
                    merged_payload.pop("float_time_stamp", None)
                    self.vector_store.overwrite_payload(
                        point_id=newest_id,
                        payload=merged_payload,
                        vector=new_embedding,
                    )
                    
                    # 删除其他记忆
                    for entry in cluster[1:]:
                        try:
                            self.vector_store.delete_one(entry["id"])
                            with lock:
                                deleted_count += 1
                        except Exception as e:
                            logger.error(f"[{call_id}] 删除记忆 {entry['id']} 失败: {e}")
                    
                    with lock:
                        merged_count += 1
                    
                    logger.debug(f"[{call_id}] 合并成功: {len(cluster)} 条 -> 1 条")
                    
                except Exception as e:
                    logger.error(f"[{call_id}] 更新记忆 {newest_id} 失败: {e}")
                    with lock:
                        failed_count += 1
        
        logger.info(f"[{call_id}] 开始并行合并记忆，使用 {max_workers} 个 worker")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(merge_cluster, clusters), total=len(clusters), desc="合并相似记忆"))
        
        stats = {
            "clusters": len(clusters),
            "merged": merged_count,
            "deleted": deleted_count,
            "failed": failed_count
        }
        
        logger.info(f"[{call_id}] 合并完成:")
        logger.info(f"[{call_id}]   - 聚类数: {stats['clusters']}")
        logger.info(f"[{call_id}]   - 成功合并: {stats['merged']} 组")
        logger.info(f"[{call_id}]   - 删除记忆: {stats['deleted']} 条")
        logger.info(f"[{call_id}]   - 失败: {stats['failed']} 组")
        logger.info(f"========== END {call_id} ==========")
        
        return stats



#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""面向 MemoryItem 的 Qdrant 向量存储客户端（bge-m3: dense + sparse 命名向量）。

Collection 设计：
- dense 向量: 名为 "dense", size = config.qdrant.DENSE_DIM, distance = cosine
- sparse 向量: 名为 "sparse"（bge-m3 lexical weights）
- payload: 与 README 描述对齐
    {
        memory_id, user_id, memory_content,
        memory_type, memory_category, status,
        importance, confidence, content_hash,
        created_at, updated_at
    }
"""

from __future__ import annotations

# 允许 `python <this_file>` 直接执行: 把项目根 (memA/) 加进 sys.path.
# 仅当作为脚本独立运行时生效, 被其他模块 import 时不会进入该分支.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models

from config import qdrant_schema
from config.setting import Config, QdrantConfig


@dataclass
class SparseVector:
    indices: List[int] = field(default_factory=list)
    values: List[float] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.indices or not self.values


class QdrantMemoryStoreException(Exception):
    pass


class QdrantMemoryStore:
    """MemoryItem 在 Qdrant 中的向量存储（dense + sparse 命名向量）。

    Collection schema(vector 命名 + payload 索引)在 :mod:`config.qdrant_schema`
    中集中维护, 本类只负责把字符串 schema_type 映射为 qdrant 枚举.
    """

    DENSE_NAME = qdrant_schema.DENSE_VECTOR_NAME
    SPARSE_NAME = qdrant_schema.SPARSE_VECTOR_NAME

    # 任意业务字符串 -> 稳定派生 UUID 的 namespace.
    # 使用 RFC 内置 NAMESPACE_OID, 同一字符串永远映射到同一 UUID.
    _POINT_ID_NAMESPACE = uuid.NAMESPACE_OID

    # 把 config 层的字符串 schema_type 解析回 qdrant 枚举
    _PAYLOAD_SCHEMA_TYPE_MAP = {
        qdrant_schema.KEYWORD: models.PayloadSchemaType.KEYWORD,
        qdrant_schema.DATETIME: models.PayloadSchemaType.DATETIME,
        qdrant_schema.FLOAT: models.PayloadSchemaType.FLOAT,
        qdrant_schema.INTEGER: models.PayloadSchemaType.INTEGER,
        qdrant_schema.BOOL: models.PayloadSchemaType.BOOL,
        qdrant_schema.TEXT: models.PayloadSchemaType.TEXT,
    }

    def __init__(self, cfg: Optional[QdrantConfig] = None):
        self.cfg = cfg or Config.qdrant
        self.collection_name = self.cfg.COLLECTION
        self.dense_dim = int(self.cfg.DENSE_DIM)
        self.distance = self._resolve_distance(self.cfg.DISTANCE)

        self.client = QdrantClient(
            url=self.cfg.URL,
            api_key=self.cfg.API_KEY or None,
            timeout=int(self.cfg.TIMEOUT),
        )
        self.ensure_collection()

    # ------------------------------------------------------------------
    # collection 生命周期
    # ------------------------------------------------------------------
    def ensure_collection(self) -> None:
        try:
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        self.DENSE_NAME: models.VectorParams(
                            size=self.dense_dim,
                            distance=self.distance,
                        )
                    },
                    sparse_vectors_config={
                        self.SPARSE_NAME: models.SparseVectorParams()
                    },
                )
                logger.info(
                    f"[QdrantMemoryClient] 创建 collection 成功: {self.collection_name} "
                    f"(dense_dim={self.dense_dim})"
                )

            for field_name, schema_type in qdrant_schema.PAYLOAD_INDEXES:
                resolved = self._PAYLOAD_SCHEMA_TYPE_MAP.get(schema_type)
                if resolved is None:
                    logger.warning(
                        f"[QdrantMemoryClient] 未识别的 payload schema_type "
                        f"{schema_type!r} (field={field_name}), 已跳过"
                    )
                    continue
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema=resolved,
                        wait=True,
                    )
                except Exception:
                    # 索引已存在或服务忽略
                    pass
        except Exception as e:
            logger.error(f"[QdrantMemoryClient] ensure_collection 失败: {e}")
            raise QdrantMemoryStoreException(str(e))

    def drop_collection(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception as e:  # pragma: no cover
            logger.warning(f"drop_collection 失败: {e}")

    # ------------------------------------------------------------------
    # upsert
    # ------------------------------------------------------------------
    def upsert_one(
        self,
        point_id: Union[str, int, uuid.UUID],
        dense_vector: Sequence[float],
        sparse_vector: Optional[SparseVector],
        payload: Dict[str, Any],
    ) -> None:
        self.upsert_batch(
            [
                {
                    "id": point_id,
                    "dense": list(dense_vector),
                    "sparse": sparse_vector,
                    "payload": payload,
                }
            ]
        )

    def upsert_batch(self, points: Iterable[Dict[str, Any]]) -> int:
        struct_points: List[models.PointStruct] = []
        for point in points:
            vector: Dict[str, Any] = {self.DENSE_NAME: list(point["dense"])}
            sparse = point.get("sparse")
            if sparse is not None and not (isinstance(sparse, SparseVector) and sparse.is_empty()):
                if isinstance(sparse, SparseVector):
                    vector[self.SPARSE_NAME] = models.SparseVector(
                        indices=list(sparse.indices),
                        values=[float(v) for v in sparse.values],
                    )
                elif isinstance(sparse, dict):
                    vector[self.SPARSE_NAME] = models.SparseVector(
                        indices=list(sparse.get("indices", [])),
                        values=[float(v) for v in sparse.get("values", [])],
                    )
            struct_points.append(
                models.PointStruct(
                    id=self._normalize_point_id(point["id"]),
                    vector=vector,
                    payload=point.get("payload", {}),
                )
            )

        if not struct_points:
            return 0

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=struct_points,
                wait=True,
            )
        except Exception as e:
            logger.error(f"[QdrantMemoryClient] upsert 失败: {e}")
            raise QdrantMemoryStoreException(str(e))
        return len(struct_points)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def search_dense(
        self,
        query_dense: Sequence[float],
        limit: int = 10,
        payload_filter: Optional[models.Filter] = None,
        score_threshold: Optional[float] = None,
        with_vectors: bool = False,
    ) -> List[Dict[str, Any]]:
        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=list(query_dense),
                using=self.DENSE_NAME,
                query_filter=payload_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=with_vectors,
            )
        except Exception as e:
            logger.error(f"[QdrantMemoryClient] search_dense 失败: {e}")
            raise QdrantMemoryStoreException(str(e))
        return [self._format_hit(h, with_vectors=with_vectors) for h in response.points]

    def search_sparse(
        self,
        sparse: SparseVector,
        limit: int = 10,
        payload_filter: Optional[models.Filter] = None,
        with_vectors: bool = False,
    ) -> List[Dict[str, Any]]:
        if sparse.is_empty():
            return []
        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=models.SparseVector(
                    indices=list(sparse.indices),
                    values=[float(v) for v in sparse.values],
                ),
                using=self.SPARSE_NAME,
                query_filter=payload_filter,
                limit=limit,
                with_payload=True,
                with_vectors=with_vectors,
            )
        except Exception as e:
            logger.error(f"[QdrantMemoryClient] search_sparse 失败: {e}")
            raise QdrantClientException(str(e))
        return [self._format_hit(h, with_vectors=with_vectors) for h in response.points]

    def scroll_all(
        self,
        payload_filter: Optional[models.Filter] = None,
        limit: int = 256,
        with_vectors: bool = False,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        offset = None
        try:
            while True:
                points, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=payload_filter,
                    limit=limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=with_vectors,
                )
                results.extend(
                    self._format_record(p, with_vectors=with_vectors) for p in points
                )
                if offset is None:
                    break
        except Exception as e:
            logger.error(f"[QdrantMemoryClient] scroll 失败: {e}")
            raise QdrantClientException(str(e))
        return results

    def get_by_ids(
        self,
        ids: Sequence[str],
        with_vectors: bool = False,
    ) -> List[Dict[str, Any]]:
        if not ids:
            return []
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[self._normalize_point_id(item) for item in ids],
                with_payload=True,
                with_vectors=with_vectors,
            )
        except Exception as e:
            logger.error(f"[QdrantMemoryClient] retrieve 失败: {e}")
            raise QdrantClientException(str(e))
        return [self._format_record(p, with_vectors=with_vectors) for p in points]

    def set_payload(self, point_id: Union[str, int, uuid.UUID], payload: Dict[str, Any]) -> None:
        try:
            self.client.set_payload(
                collection_name=self.collection_name,
                payload=payload,
                points=[self._normalize_point_id(point_id)],
                wait=True,
            )
        except Exception as e:
            logger.error(f"[QdrantMemoryClient] set_payload 失败: {e}")
            raise QdrantClientException(str(e))

    def delete(self, point_ids: Sequence[Union[str, int, uuid.UUID]]) -> None:
        if not point_ids:
            return
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[self._normalize_point_id(pid) for pid in point_ids]
                ),
                wait=True,
            )
        except Exception as e:
            logger.error(f"[QdrantMemoryClient] delete 失败: {e}")
            raise QdrantClientException(str(e))

    def count(self, payload_filter: Optional[models.Filter] = None) -> int:
        try:
            response = self.client.count(
                collection_name=self.collection_name,
                count_filter=payload_filter,
                exact=True,
            )
        except Exception as e:
            logger.error(f"[QdrantMemoryClient] count 失败: {e}")
            raise QdrantClientException(str(e))
        return response.count

    # ------------------------------------------------------------------
    # 过滤器工厂
    # ------------------------------------------------------------------
    @classmethod
    def build_filter(
        cls,
        user_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        memory_category: Optional[str] = None,
        status: Optional[str] = "active",
        updated_at_gte: Optional[str] = None,
        updated_at_lte: Optional[str] = None,
        content_hash: Optional[str] = None,
        memory_ids: Optional[Sequence[str]] = None,
        extra_must: Optional[List[models.Condition]] = None,
    ) -> Optional[models.Filter]:
        must: List[models.Condition] = []
        if user_id:
            must.append(
                models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))
            )
        if memory_type:
            must.append(
                models.FieldCondition(
                    key="memory_type", match=models.MatchValue(value=memory_type)
                )
            )
        if memory_category:
            must.append(
                models.FieldCondition(
                    key="memory_category",
                    match=models.MatchValue(value=memory_category),
                )
            )
        if status:
            must.append(
                models.FieldCondition(key="status", match=models.MatchValue(value=status))
            )
        if content_hash:
            must.append(
                models.FieldCondition(
                    key="content_hash", match=models.MatchValue(value=content_hash)
                )
            )
        if memory_ids:
            must.append(
                models.FieldCondition(
                    key="memory_id",
                    match=models.MatchAny(any=[str(m) for m in memory_ids]),
                )
            )
        if updated_at_gte or updated_at_lte:
            must.append(
                models.FieldCondition(
                    key="updated_at",
                    range=models.DatetimeRange(gte=updated_at_gte, lte=updated_at_lte),
                )
            )
        if extra_must:
            must.extend(extra_must)
        return models.Filter(must=must) if must else None

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    @staticmethod
    def sort_records(
        records: List[Dict[str, Any]],
        key: str,
        reverse: bool = False,
    ) -> List[Dict[str, Any]]:
        def sort_key(item: Dict[str, Any]) -> Any:
            value = item.get(key)
            if value is None:
                return ""
            if isinstance(value, str):
                try:
                    return datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError:
                    return value
            return value

        return sorted(records, key=sort_key, reverse=reverse)

    @classmethod
    def _normalize_point_id(cls, point_id: Union[str, int, uuid.UUID]) -> Union[int, str]:
        """把任意业务 id 规范化为 Qdrant 允许的 point id.

        Qdrant 只接受 unsigned int 或 UUID 字符串. 规则:
            - ``int`` (非负) -> 原样
            - 合法 UUID 字符串 / ``uuid.UUID`` -> ``str(uuid)``
            - 其他任意字符串 -> ``uuid.uuid5(NAMESPACE_OID, s)``, 确定性派生

        业务侧自定义 id (如 ``mem_abc123``) 因此可以无感知地直接传, 同一
        字符串永远映射到同一 UUID, upsert / get_by_ids / delete 全部对得上.
        业务 id 原值应另行存储在 payload (例如 ``memory_id``) 字段.
        """
        if isinstance(point_id, uuid.UUID):
            return str(point_id)
        if isinstance(point_id, bool):  # bool 是 int 的子类, 但语义不对
            raise TypeError(f"point_id 不能为 bool: {point_id!r}")
        if isinstance(point_id, int):
            if point_id < 0:
                raise ValueError(f"point_id 必须为非负整数, got {point_id}")
            return point_id
        s = str(point_id)
        try:
            return str(uuid.UUID(s))
        except ValueError:
            return str(uuid.uuid5(cls._POINT_ID_NAMESPACE, s))

    @staticmethod
    def _resolve_distance(distance: str) -> models.Distance:
        normalized = (distance or "cosine").lower()
        mapping = {
            "cosine": models.Distance.COSINE,
            "dot": models.Distance.DOT,
            "dot_product": models.Distance.DOT,
            "euclid": models.Distance.EUCLID,
            "euclidean": models.Distance.EUCLID,
            "manhattan": models.Distance.MANHATTAN,
        }
        return mapping.get(normalized, models.Distance.COSINE)

    @classmethod
    def _format_hit(cls, hit: Any, with_vectors: bool = False) -> Dict[str, Any]:
        payload = dict(hit.payload or {})
        payload["_id"] = str(hit.id)
        payload["_score"] = float(hit.score)
        if with_vectors:
            payload["_vectors"] = cls._normalize_vectors(getattr(hit, "vector", None))
        return payload

    @classmethod
    def _format_record(cls, point: Any, with_vectors: bool = False) -> Dict[str, Any]:
        payload = dict(point.payload or {})
        payload["_id"] = str(point.id)
        if with_vectors:
            payload["_vectors"] = cls._normalize_vectors(getattr(point, "vector", None))
        return payload

    @classmethod
    def _normalize_vectors(cls, vectors: Any) -> Dict[str, Any]:
        if vectors is None:
            return {}
        if isinstance(vectors, dict):
            normalized: Dict[str, Any] = {}
            for name, vec in vectors.items():
                if hasattr(vec, "indices") and hasattr(vec, "values"):
                    normalized[name] = {
                        "indices": list(vec.indices),
                        "values": [float(v) for v in vec.values],
                    }
                else:
                    normalized[name] = list(vec) if vec is not None else []
            return normalized
        # 老接口直接是 list
        return {cls.DENSE_NAME: list(vectors)}

if __name__ == "__main__":
    client = QdrantMemoryStore()

    business_id = "mem_005"

    dense_dim = client.dense_dim
    dense_vec = [0.05] * dense_dim

    client.upsert_one(
        point_id=business_id,
        dense_vector=dense_vec,
        sparse_vector=SparseVector(indices=[0, 1, 2], values=[0.1, 0.2, 0.3]),
        payload={
            "user_id": "u_001",
            "memory_id": business_id,
            "memory_content": "用户是一个喜欢穿黑丝的女孩",
            "memory_type": "profile",
            "memory_category": "fact",
            "status": "active",
            "created_at": "2021-01-01T12:00:00Z",
            "importance": 0.5,
            "confidence": 0.5,
        },
    )

    print("get_by_ids (business id):", client.get_by_ids([business_id]))
    print("count:", client.count())
    
    
    # 搜索：在 user_id 约束下检索
    user_filter = QdrantMemoryStore.build_filter(user_id="u_001", status="active")
    query_dense = [0.045] * dense_dim
    hits = client.search_dense(query_dense, limit=10, payload_filter=user_filter)
    print("search_dense (user_id=u_001):", hits)

    sparse_filter = QdrantMemoryStore.build_filter(user_id="u_001")
    sparse_hits = client.search_sparse(
        SparseVector(indices=[0, 1, 2], values=[0.1, 0.2, 0.3]),
        limit=10,
        payload_filter=sparse_filter,
    )
    print("search_sparse (user_id=u_001):", sparse_hits)
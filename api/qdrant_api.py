#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models

from config.config import Config


class QdrantClientException(Exception):
    """Qdrant 客户端异常"""


class QdrantMemoryClient:
    """面向记忆存储的 Qdrant 客户端封装。"""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        vector_size: Optional[int] = None,
        distance: Optional[str] = None,
    ) -> None:
        self.collection_name = collection_name or Config.qdrant.COLLECTION
        self.vector_size = vector_size or Config.qdrant.VECTOR_SIZE
        self.distance = distance or Config.qdrant.DISTANCE
        self.client = QdrantClient(
            url=Config.qdrant.URL,
            api_key=Config.qdrant.API_KEY or None,
            timeout=Config.qdrant.TIMEOUT,
        )
        self.ensure_collection()

    def ensure_collection(self) -> None:
        """确保 collection 存在，并建立常用 payload 索引。"""
        try:
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=self._resolve_distance(self.distance),
                    ),
                )
                logger.info(f"创建 Qdrant collection 成功: {self.collection_name}")

            for field_name, schema in (
                ("child_id", models.PayloadSchemaType.KEYWORD),
                ("agent_id", models.PayloadSchemaType.KEYWORD),
                ("memory_id", models.PayloadSchemaType.KEYWORD),
                ("memory_type", models.PayloadSchemaType.KEYWORD),
                ("updated_at", models.PayloadSchemaType.DATETIME),
                ("created_at", models.PayloadSchemaType.DATETIME),
                ("merged", models.PayloadSchemaType.BOOL),
            ):
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema=schema,
                        wait=True,
                    )
                except Exception:
                    # 索引已存在时直接忽略。
                    pass
        except Exception as e:
            logger.error(f"Qdrant 初始化失败: {e}")
            raise QdrantClientException(f"Qdrant 初始化失败: {e}")

    def upsert_one(
        self,
        point_id: str,
        vector: List[float],
        payload: Dict[str, Any],
    ) -> None:
        self.upsert_batch(
            [{"id": point_id, "vector": vector, "payload": payload}],
        )

    def upsert_batch(self, points: Iterable[Dict[str, Any]]) -> None:
        qdrant_points: List[models.PointStruct] = []
        for point in points:
            qdrant_points.append(
                models.PointStruct(
                    id=str(point["id"]),
                    vector=point["vector"],
                    payload=point.get("payload", {}),
                )
            )

        if not qdrant_points:
            return

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=qdrant_points,
                wait=True,
            )
        except Exception as e:
            logger.error(f"Qdrant upsert 失败: {e}")
            raise QdrantClientException(f"Qdrant upsert 失败: {e}")

    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        payload_filter: Optional[models.Filter] = None,
        score_threshold: Optional[float] = None,
        with_vectors: bool = False,
    ) -> List[Dict[str, Any]]:
        try:
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=payload_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=with_vectors,
            )
        except Exception as e:
            logger.error(f"Qdrant search 失败: {e}")
            raise QdrantClientException(f"Qdrant search 失败: {e}")

        return [self._format_hit(hit, with_vectors=with_vectors) for hit in hits]

    def scroll_all(
        self,
        payload_filter: Optional[models.Filter] = None,
        limit: int = 256,
        with_vectors: bool = False,
    ) -> List[Dict[str, Any]]:
        all_points: List[Dict[str, Any]] = []
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
                all_points.extend(
                    self._format_record(point, with_vectors=with_vectors) for point in points
                )
                if offset is None:
                    break
        except Exception as e:
            logger.error(f"Qdrant scroll 失败: {e}")
            raise QdrantClientException(f"Qdrant scroll 失败: {e}")

        return all_points

    def get_by_ids(
        self,
        ids: List[str],
        with_vectors: bool = False,
    ) -> List[Dict[str, Any]]:
        if not ids:
            return []
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[str(item) for item in ids],
                with_payload=True,
                with_vectors=with_vectors,
            )
        except Exception as e:
            logger.error(f"Qdrant retrieve 失败: {e}")
            raise QdrantClientException(f"Qdrant retrieve 失败: {e}")
        return [self._format_record(point, with_vectors=with_vectors) for point in points]

    def set_payload(
        self,
        point_id: str,
        payload: Dict[str, Any],
    ) -> None:
        try:
            self.client.set_payload(
                collection_name=self.collection_name,
                payload=payload,
                points=[str(point_id)],
                wait=True,
            )
        except Exception as e:
            logger.error(f"Qdrant set_payload 失败: {e}")
            raise QdrantClientException(f"Qdrant set_payload 失败: {e}")

    def overwrite_payload(
        self,
        point_id: str,
        payload: Dict[str, Any],
        vector: Optional[List[float]] = None,
    ) -> None:
        if vector is None:
            records = self.get_by_ids([point_id], with_vectors=True)
            if not records:
                raise QdrantClientException(f"Point 不存在: {point_id}")
            vector = records[0].get("vector")
        self.upsert_one(point_id=point_id, vector=vector, payload=payload)

    def delete_one(self, point_id: str) -> None:
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=[str(point_id)]),
                wait=True,
            )
        except Exception as e:
            logger.error(f"Qdrant delete 失败: {e}")
            raise QdrantClientException(f"Qdrant delete 失败: {e}")

    def delete_by_filter(self, payload_filter: models.Filter) -> None:
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=payload_filter,
                wait=True,
            )
        except Exception as e:
            logger.error(f"Qdrant filter delete 失败: {e}")
            raise QdrantClientException(f"Qdrant filter delete 失败: {e}")

    def count(self, payload_filter: Optional[models.Filter] = None) -> int:
        try:
            response = self.client.count(
                collection_name=self.collection_name,
                count_filter=payload_filter,
                exact=True,
            )
        except Exception as e:
            logger.error(f"Qdrant count 失败: {e}")
            raise QdrantClientException(f"Qdrant count 失败: {e}")
        return response.count

    @staticmethod
    def build_filter(
        child_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        updated_at_gte: Optional[str] = None,
        updated_at_lte: Optional[str] = None,
        extra_must: Optional[List[models.Condition]] = None,
    ) -> Optional[models.Filter]:
        must: List[models.Condition] = []

        if child_id:
            must.append(
                models.FieldCondition(
                    key="child_id",
                    match=models.MatchValue(value=child_id),
                )
            )
        if agent_id:
            must.append(
                models.FieldCondition(
                    key="agent_id",
                    match=models.MatchValue(value=agent_id),
                )
            )
        if memory_type:
            must.append(
                models.FieldCondition(
                    key="memory_type",
                    match=models.MatchValue(value=memory_type),
                )
            )
        if updated_at_gte or updated_at_lte:
            must.append(
                models.FieldCondition(
                    key="updated_at",
                    range=models.DatetimeRange(
                        gte=updated_at_gte,
                        lte=updated_at_lte,
                    ),
                )
            )
        if extra_must:
            must.extend(extra_must)

        if not must:
            return None
        return models.Filter(must=must)

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

    def _format_hit(
        self,
        hit: Any,
        with_vectors: bool = False,
    ) -> Dict[str, Any]:
        payload = dict(hit.payload or {})
        payload["_id"] = str(hit.id)
        payload["_score"] = float(hit.score)
        if with_vectors:
            payload["vector"] = self._normalize_vector(hit.vector)
        return payload

    def _format_record(
        self,
        point: Any,
        with_vectors: bool = False,
    ) -> Dict[str, Any]:
        payload = dict(point.payload or {})
        payload["_id"] = str(point.id)
        if with_vectors:
            payload["vector"] = self._normalize_vector(point.vector)
        return payload

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

    @staticmethod
    def _normalize_vector(vector: Any) -> List[float]:
        if vector is None:
            return []
        if isinstance(vector, dict):
            first_value = next(iter(vector.values()), [])
            return list(first_value or [])
        return list(vector)

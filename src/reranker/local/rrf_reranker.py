#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reciprocal Rank Fusion reranker for memory items."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


class RRFReranker:
    """Fuse multiple ranked memory lists with Reciprocal Rank Fusion.

    RRF score formula:
        score = sum(weight / (rank_constant + rank))

    ``rank`` starts from 1 in each source list. Dict inputs are copied and
    enriched with ``rrf_rank``, ``rrf_score`` and ``rrf_sources``.
    """

    def __init__(
        self,
        rank_constant: int = 60,
        id_keys: Optional[Sequence[str]] = None,
    ):
        if rank_constant < 1:
            raise ValueError("rank_constant must be >= 1")

        self.rank_constant = rank_constant
        self.id_keys = tuple(id_keys or ("memory_id", "id", "_id"))

    def rerank(
        self,
        ranked_lists: Sequence[Sequence[Any]],
        top_k: int = 10,
        weights: Optional[Sequence[float]] = None,
    ) -> List[Any]:
        """Return fused memory items ordered by RRF score."""
        if not ranked_lists:
            return []

        top_k = max(1, top_k)
        weights = self._normalize_weights(ranked_lists, weights)
        scores: Dict[str, float] = {}
        best_items: Dict[str, Tuple[Any, int, int]] = {}
        sources: Dict[str, List[Dict[str, Any]]] = {}

        for list_index, memory_items in enumerate(ranked_lists):
            seen_in_list = set()
            for zero_based_rank, item in enumerate(memory_items):
                key = self._item_key(item)
                if not key or key in seen_in_list:
                    continue

                seen_in_list.add(key)
                rank = zero_based_rank + 1
                score = weights[list_index] / (self.rank_constant + rank)
                scores[key] = scores.get(key, 0.0) + score
                sources.setdefault(key, []).append(
                    {
                        "list_index": list_index,
                        "rank": rank,
                        "weight": weights[list_index],
                        "score": score,
                    }
                )

                current_best = best_items.get(key)
                if current_best is None or rank < current_best[1]:
                    best_items[key] = (item, rank, list_index)

        ordered_keys = sorted(
            scores,
            key=lambda key: (
                -scores[key],
                best_items[key][1],
                best_items[key][2],
            ),
        )

        results = []
        for rank, key in enumerate(ordered_keys[:top_k], start=1):
            item = best_items[key][0]
            results.append(
                self._attach_rrf_fields(
                    item=item,
                    rank=rank,
                    score=scores[key],
                    sources=sources[key],
                )
            )
        return results

    def _normalize_weights(
        self,
        ranked_lists: Sequence[Sequence[Any]],
        weights: Optional[Sequence[float]],
    ) -> List[float]:
        if weights is None:
            return [1.0] * len(ranked_lists)
        if len(weights) != len(ranked_lists):
            raise ValueError("weights length must match ranked_lists length")
        return [float(weight) for weight in weights]

    def _item_key(self, item: Any) -> str:
        data = self._to_dict(item)
        for key in self.id_keys:
            value = data.get(key)
            if value:
                return str(value)

        content = data.get("memory_content") or data.get("content")
        if content:
            return f"content:{content}"
        return str(item)

    @staticmethod
    def _to_dict(item: Any) -> Dict[str, Any]:
        if isinstance(item, dict):
            return item
        if is_dataclass(item):
            return asdict(item)
        if hasattr(item, "__dict__"):
            return dict(item.__dict__)
        return {"memory_content": str(item)}

    @staticmethod
    def _attach_rrf_fields(
        item: Any,
        rank: int,
        score: float,
        sources: List[Dict[str, Any]],
    ) -> Any:
        if not isinstance(item, dict):
            return item

        ranked_item = deepcopy(item)
        ranked_item["rrf_rank"] = rank
        ranked_item["rrf_score"] = score
        ranked_item["rrf_sources"] = sources
        return ranked_item


def rrf_rerank(
    ranked_lists: Sequence[Sequence[Any]],
    top_k: int = 10,
    rank_constant: int = 60,
    weights: Optional[Sequence[float]] = None,
    id_keys: Optional[Sequence[str]] = None,
) -> List[Any]:
    """Convenience function for RRF reranking memory items."""
    return RRFReranker(
        rank_constant=rank_constant,
        id_keys=id_keys,
    ).rerank(
        ranked_lists=ranked_lists,
        top_k=top_k,
        weights=weights,
    )

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM reranker for recalled memory items."""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from loguru import logger


SYSTEM_PROMPT = """你是一个记忆检索排序器。
你的任务是根据用户当前 query，对候选 memory_item 做相关性排序。
只返回 JSON，不要输出解释性文本。"""

USER_PROMPT = """用户 query：
{query}

候选 memory_item：
{candidates}

请按以下规则排序：
1. 优先选择能直接回答或支撑 query 的记忆。
2. 同等相关时，优先选择更具体、更新、更可信的记忆。
3. 不相关或弱相关的记忆可以排在后面，但不要编造候选中不存在的内容。

输出 JSON object，格式必须为：
{{
  "ranked": [
    {{"index": 0, "score": 0.95, "reason": "简短理由"}}
  ]
}}

要求：
- index 必须来自候选 memory_item 的 index。
- score 是 0 到 1 的相关性分数。
- ranked 最多返回 {top_k} 条。"""


class LLMReranker:
    """Use an LLM to rerank recalled memory items.

    The public API intentionally accepts ``Dict`` records because online
    retrieval currently returns dictionaries, while also supporting dataclass
    ``MemoryItem`` objects for offline or typed callers.
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        config: Optional[Any] = None,
        max_candidates: int = 40,
        max_content_chars: int = 500,
    ):
        self.llm = llm or self._build_default_llm(config)
        self.max_candidates = max(1, max_candidates)
        self.max_content_chars = max(80, max_content_chars)

    def rerank(
        self,
        query: str,
        memory_items: Sequence[Any],
        top_k: int = 10,
    ) -> List[Any]:
        """Return memory items ordered by LLM relevance.

        The returned item type matches the input item type. Dict inputs are
        copied and enriched with ``llm_rank``, ``llm_score`` and
        ``llm_reason``. Dataclass inputs are returned as the original objects
        without mutation because their schema has no rerank fields.
        """
        if not query or not memory_items:
            return list(memory_items[:top_k])

        candidates = list(memory_items[: self.max_candidates])
        top_k = min(max(1, top_k), len(candidates))

        try:
            payload = self._format_candidates(candidates)
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": USER_PROMPT.format(
                            query=query,
                            candidates=json.dumps(payload, ensure_ascii=False, indent=2),
                            top_k=top_k,
                        ),
                    },
                ],
                response_format={"type": "json_object"},
            )
            ranked = self._parse_response(response, len(candidates))
            if not ranked:
                return self._fallback(candidates, top_k)

            return self._build_ranked_items(candidates, ranked, top_k)
        except Exception as exc:
            logger.warning(f"LLM rerank 失败，返回原始召回顺序: {exc}")
            return self._fallback(candidates, top_k)

    @staticmethod
    def _build_default_llm(config: Optional[Any] = None) -> Any:
        from config.setting import Config as _Config
        from src.llm.openai_llm import LLMApi

        cfg = config or _Config
        return LLMApi.from_params(
            api_key=cfg.llm.API_KEY,
            base_url=cfg.llm.BASE_URL,
            model=cfg.llm.MODEL,
            temperature=0,
            max_tokens=min(getattr(cfg.llm, "MAX_TOKENS", 2048), 2048),
            top_p=getattr(cfg.llm, "TOP_P", 0.95),
            timeout=getattr(cfg.llm, "TIMEOUT", 60),
        )

    def _format_candidates(self, memory_items: Sequence[Any]) -> List[Dict[str, Any]]:
        payload = []
        for index, item in enumerate(memory_items):
            data = self._to_dict(item)
            content = str(data.get("memory_content") or data.get("content") or "")
            payload.append(
                {
                    "index": index,
                    "memory_id": data.get("memory_id") or data.get("id") or data.get("_id"),
                    "memory_content": content[: self.max_content_chars],
                    "memory_type": data.get("memory_type", ""),
                    "memory_category": data.get("memory_category", ""),
                    "updated_at": data.get("updated_at", ""),
                    "score": data.get("score") if data.get("score") is not None else data.get("_score"),
                    "retrieval_count": data.get("retrieval_count"),
                }
            )
        return payload

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
    def _parse_response(response: str, candidate_count: int) -> List[Dict[str, Any]]:
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", response, flags=re.S)
            if not match:
                return []
            result = json.loads(match.group(0))

        if isinstance(result, list):
            ranked = result
        elif isinstance(result, dict):
            ranked = result.get("ranked", [])
        else:
            ranked = []
        if not isinstance(ranked, list):
            return []

        seen = set()
        normalized = []
        for item in ranked:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item["index"])
            except (KeyError, TypeError, ValueError):
                continue
            if index < 0 or index >= candidate_count or index in seen:
                continue

            seen.add(index)
            normalized.append(
                {
                    "index": index,
                    "score": LLMReranker._safe_float(item.get("score"), default=0.0),
                    "reason": str(item.get("reason", "")),
                }
            )
        return normalized

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return default
        return max(0.0, min(1.0, score))

    def _build_ranked_items(
        self,
        candidates: Sequence[Any],
        ranked: Iterable[Dict[str, Any]],
        top_k: int,
    ) -> List[Any]:
        results = []
        used = set()

        for rank, item in enumerate(ranked, start=1):
            if len(results) >= top_k:
                break
            index = item["index"]
            used.add(index)
            results.append(
                self._attach_rank_fields(
                    candidates[index],
                    rank=rank,
                    score=item["score"],
                    reason=item["reason"],
                )
            )

        for index, candidate in enumerate(candidates):
            if len(results) >= top_k:
                break
            if index in used:
                continue
            results.append(self._attach_rank_fields(candidate, rank=len(results) + 1))

        return results

    @staticmethod
    def _attach_rank_fields(
        item: Any,
        rank: int,
        score: Optional[float] = None,
        reason: str = "",
    ) -> Any:
        if not isinstance(item, dict):
            return item

        ranked_item = deepcopy(item)
        ranked_item["llm_rank"] = rank
        if score is not None:
            ranked_item["llm_score"] = score
        if reason:
            ranked_item["llm_reason"] = reason
        return ranked_item

    @staticmethod
    def _fallback(memory_items: Sequence[Any], top_k: int) -> List[Any]:
        return list(memory_items[:top_k])


def rerank_memory_items(
    query: str,
    memory_items: Sequence[Any],
    top_k: int = 10,
    llm: Optional[Any] = None,
    config: Optional[Any] = None,
) -> List[Any]:
    """Convenience function for reranking recalled memory items."""
    return LLMReranker(llm=llm, config=config).rerank(
        query=query,
        memory_items=memory_items,
        top_k=top_k,
    )

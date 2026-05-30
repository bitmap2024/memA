#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BgeReranker：基于 bge-reranker-v2-m3 cross-encoder 对召回结果做精排。

- 若本地模型加载成功，使用 transformers AutoModelForSequenceClassification 推理。
- 若不可用（torch 未安装、模型路径不存在等），自动退化为基于 dense 召回 score 的排序，
  保证管线在任何环境下都不阻塞。
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence

from loguru import logger

from config.setting import Config


class BgeReranker:
    """bge-reranker-v2-m3 cross-encoder 包装。"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        self.model_path = model_path or Config.reranker.BGE_RERANK_PATH
        self.device = device or Config.reranker.BGE_RERANK_DEVICE
        self.max_length = int(max_length)
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._enabled = self._try_load()

    def _try_load(self) -> bool:
        if not self.model_path or not os.path.exists(self.model_path):
            logger.info(
                f"[BgeReranker] 模型路径不存在: {self.model_path}，退化为分数排序"
            )
            return False
        try:  # pragma: no cover - optional heavy deps
            import torch  # type: ignore
            from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self._model.to(self.device)
            self._model.eval()
            logger.info(f"[BgeReranker] 模型加载成功: {self.model_path} @ {self.device}")
            return True
        except Exception as e:  # pragma: no cover
            logger.warning(f"[BgeReranker] 加载失败，退化为分数排序: {e}")
            return False

    # ------------------------------------------------------------------
    # 排序
    # ------------------------------------------------------------------
    def rerank(
        self,
        query: str,
        candidates: Sequence[Dict[str, Any]],
        top_k: int = 10,
        content_key: str = "memory_content",
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        top_k = max(1, top_k)
        if not self._enabled:
            ordered = sorted(
                candidates,
                key=lambda x: float(x.get("dense_score") or x.get("score") or 0.0),
                reverse=True,
            )
            return [
                self._attach(item, rank=i + 1, score=float(item.get("score") or 0.0))
                for i, item in enumerate(ordered[:top_k])
            ]

        pairs = [(query, str(c.get(content_key) or "")) for c in candidates]
        scores = self._score_pairs(pairs)
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            self._attach(deepcopy(item), rank=i + 1, score=float(score))
            for i, (item, score) in enumerate(scored[:top_k])
        ]

    def _score_pairs(self, pairs: List[tuple]) -> List[float]:  # pragma: no cover
        torch = self._torch
        with torch.no_grad():
            inputs = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            logits = self._model(**inputs).logits.view(-1).float()
            scores = torch.sigmoid(logits)
            return [float(s) for s in scores.cpu().tolist()]

    @staticmethod
    def _attach(item: Any, rank: int, score: float) -> Any:
        if not isinstance(item, dict):
            return item
        item = dict(item)
        item["bge_rerank_rank"] = rank
        item["bge_rerank_score"] = score
        return item

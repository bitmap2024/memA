#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""轻量 BM25 检索器（在用户级活跃记忆全集合上构建倒排，本地内存即可）。"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from config.releatiion_schema import MemoryItem


_WORD_RE = re.compile(r"[A-Za-z]+|[0-9]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return _WORD_RE.findall(str(text).lower())


@dataclass
class _IndexEntry:
    memory_id: str
    length: int
    counts: Counter


class BM25Searcher:
    """非常轻量的 BM25，按 user_id 构建索引，按需重建。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = float(k1)
        self.b = float(b)
        self._lock = threading.RLock()
        self._user_index: Dict[str, Dict[str, _IndexEntry]] = {}
        self._user_avgdl: Dict[str, float] = {}
        self._user_df: Dict[str, Counter] = {}

    def build_for_user(self, user_id: str, memories: Sequence[MemoryItem]) -> None:
        with self._lock:
            index: Dict[str, _IndexEntry] = {}
            df: Counter = Counter()
            total_len = 0
            for mem in memories:
                tokens = tokenize(mem.memory_content)
                if not tokens:
                    continue
                counts = Counter(tokens)
                index[mem.memory_id] = _IndexEntry(
                    memory_id=mem.memory_id, length=len(tokens), counts=counts
                )
                total_len += len(tokens)
                for term in counts:
                    df[term] += 1
            avgdl = (total_len / len(index)) if index else 0.0
            self._user_index[user_id] = index
            self._user_avgdl[user_id] = avgdl
            self._user_df[user_id] = df

    def has_index(self, user_id: str) -> bool:
        return user_id in self._user_index

    def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, float]]:
        with self._lock:
            index = self._user_index.get(user_id) or {}
            avgdl = self._user_avgdl.get(user_id) or 0.0
            df = self._user_df.get(user_id) or Counter()

        if not index:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []

        N = len(index)
        scores: Dict[str, float] = defaultdict(float)
        for term in tokens:
            term_df = df.get(term, 0)
            if term_df == 0:
                continue
            idf = math.log(1 + (N - term_df + 0.5) / (term_df + 0.5))
            for memory_id, entry in index.items():
                term_freq = entry.counts.get(term, 0)
                if not term_freq:
                    continue
                denom = term_freq + self.k1 * (1 - self.b + self.b * (entry.length / (avgdl or 1)))
                scores[memory_id] += idf * ((term_freq * (self.k1 + 1)) / (denom or 1))

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [{"memory_id": mid, "score": float(score)} for mid, score in ranked]

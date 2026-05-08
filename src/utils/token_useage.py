#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM token 使用量统计工具。
"""

from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, Optional


@dataclass
class TokenUsage:
    """单次或累计的 token 使用量。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self):
        self.prompt_tokens = max(0, int(self.prompt_tokens or 0))
        self.completion_tokens = max(0, int(self.completion_tokens or 0))
        self.total_tokens = max(0, int(self.total_tokens or 0))
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens

    def add(self, usage: "TokenUsage") -> "TokenUsage":
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens
        return self

    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


class TokenUsageTracker:
    """
    记录 LLM 调用的 token 使用量，支持总量和按 model 分组统计。

    用法:
        tracker = TokenUsageTracker()
        tracker.add(prompt_tokens=100, completion_tokens=20, model="qwen-turbo")
        tracker.add_response(response, model="qwen-turbo")
        print(tracker.summary())
    """

    def __init__(self):
        self._total = TokenUsage()
        self._by_model: Dict[str, TokenUsage] = {}
        self._calls = 0
        self._lock = Lock()

    @property
    def calls(self) -> int:
        return self._calls

    @property
    def total(self) -> TokenUsage:
        return TokenUsage(**self._total.to_dict())

    def add(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        model: Optional[str] = None,
    ) -> TokenUsage:
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        with self._lock:
            self._calls += 1
            self._total.add(usage)
            if model:
                self._by_model.setdefault(model, TokenUsage()).add(usage)

        return usage

    def add_usage(self, usage: Any, model: Optional[str] = None) -> TokenUsage:
        """
        从 OpenAI 兼容 usage 对象或 dict 中记录 token。
        """
        prompt_tokens = self._get_value(usage, "prompt_tokens")
        completion_tokens = self._get_value(usage, "completion_tokens")
        total_tokens = self._get_value(usage, "total_tokens")
        return self.add(prompt_tokens, completion_tokens, total_tokens, model)

    def add_response(self, response: Any, model: Optional[str] = None) -> TokenUsage:
        """
        从 LLM response 中提取 usage 并记录。

        兼容:
            - response.usage.prompt_tokens
            - response["usage"]["prompt_tokens"]
        """
        response_model = model or self._get_value(response, "model")
        usage = self._get_value(response, "usage")
        if usage is None:
            usage = response
        return self.add_usage(usage, response_model)

    def merge(self, other: "TokenUsageTracker") -> None:
        with self._lock:
            self._calls += other.calls
            self._total.add(other.total)
            for model, usage in other.by_model().items():
                self._by_model.setdefault(model, TokenUsage()).add(TokenUsage(**usage))

    def reset(self) -> None:
        with self._lock:
            self._total = TokenUsage()
            self._by_model = {}
            self._calls = 0

    def by_model(self) -> Dict[str, Dict[str, int]]:
        return {model: usage.to_dict() for model, usage in self._by_model.items()}

    def summary(self) -> Dict[str, Any]:
        return {
            "calls": self.calls,
            **self.total.to_dict(),
            "by_model": self.by_model(),
        }

    @staticmethod
    def _get_value(data: Any, key: str, default: Any = None) -> Any:
        if data is None:
            return default
        if isinstance(data, dict):
            return data.get(key, default)
        return getattr(data, key, default)

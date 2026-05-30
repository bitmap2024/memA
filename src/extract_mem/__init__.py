"""Memory extraction pipeline components."""

from __future__ import annotations

__all__ = ["MemoryExtractPipeline", "MultiKindLLMExtractor"]


def __getattr__(name: str):
    if name == "MemoryExtractPipeline":
        from .memory_extract_pipeline import MemoryExtractPipeline

        return MemoryExtractPipeline
    if name == "MultiKindLLMExtractor":
        from .multikind_memory_extract import MultiKindLLMExtractor

        return MultiKindLLMExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

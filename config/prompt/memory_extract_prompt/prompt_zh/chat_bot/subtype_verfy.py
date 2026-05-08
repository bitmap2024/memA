"""Memory subtype definitions and validation helpers for chat bot extraction."""

from __future__ import annotations

from typing import Mapping, Set, Tuple


PROFILE_MEMORY_TYPE = "profile"

PROFILE_MEMORY_CATEGORIES: Tuple[str, ...] = (
    "fact",
    "preference",
    "relationship",
    "goal",
    "constraint",
    "skill",
    "portrait",
    "communication_style",
)

PROFILE_MEMORY_CATEGORY_SET: Set[str] = set(PROFILE_MEMORY_CATEGORIES)

VALID_MEMORY_CATEGORIES_BY_TYPE = {
    PROFILE_MEMORY_TYPE: PROFILE_MEMORY_CATEGORY_SET,
}


def is_valid_memory_category(memory_type: str, memory_category: str) -> bool:
    """Return whether the category is allowed for the given memory type."""
    return memory_category in VALID_MEMORY_CATEGORIES_BY_TYPE.get(memory_type, set())


def is_valid_profile_memory_category(memory_category: str) -> bool:
    """Return whether the category is allowed for Profile Memory."""
    return is_valid_memory_category(PROFILE_MEMORY_TYPE, memory_category)


def validate_profile_memory(memory: Mapping[str, object]) -> bool:
    """Validate one LLM-extracted Profile Memory item."""
    return (
        memory.get("memory_type") == PROFILE_MEMORY_TYPE
        and isinstance(memory.get("memory_category"), str)
        and is_valid_profile_memory_category(memory["memory_category"])
    )


__all__ = [
    "PROFILE_MEMORY_TYPE",
    "PROFILE_MEMORY_CATEGORIES",
    "PROFILE_MEMORY_CATEGORY_SET",
    "VALID_MEMORY_CATEGORIES_BY_TYPE",
    "is_valid_memory_category",
    "is_valid_profile_memory_category",
    "validate_profile_memory",
]

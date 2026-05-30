#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""校验 LLM 抽取记忆的数据格式。

校验对象为抽取器（MultiKindLLMExtractor / UnifiedMemoryExtractor）产出的记忆条目，
单条记忆期望的标准格式为：

    {
        "topic_context": str,       # 主题摘要
        "memory_content": str,      # 非空文本
        "memory_type": str,         # profile / episodic / state
        "memory_category": str,     # 必须属于对应 memory_type 的合法子类
        "importance": float,        # [0, 1]
        "confidence": float,        # [0, 1]
        "cite": List[int],          # 来源消息索引数组
    }

提供两个层级的校验入口：
    - verify_memory_item: 校验单条记忆
    - verify_extracted_memories: 校验 extract_all 返回的 {kind: [memory, ...]} 整体结构
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


REQUIRED_FIELDS: Tuple[str, ...] = (
    "memory_content",
    "memory_type",
    "memory_category",
    "importance",
    "confidence",
    "cite",
)

VALID_MEMORY_TYPES = set(["profile", "episodic", "state"])

VALID_MEMORY_CATEGORIES_BY_TYPE = {
    "profile": set(["fact", "preference", "relationship", "goal", "constraint", "skill", "portrait", "communication_style", "instruct"]),
    "episodic": set(["conversation_event", "life_event", "project_event", "relationship_event", "decision_event", "unresolved_event"]),
    "state": set(["current_focus", "pending_followups"]),
}

def is_valid_memory_category(memory_type: str, memory_category: str) -> bool:
    return memory_category in VALID_MEMORY_CATEGORIES_BY_TYPE.get(memory_type, set())


def is_valid_profile_memory_category(memory_category: str) -> bool:
    return is_valid_memory_category("profile", memory_category)

def is_valid_episodic_memory_category(memory_category: str) -> bool:
    return is_valid_memory_category("episodic", memory_category)

def is_valid_state_memory_category(memory_category: str) -> bool:
    return is_valid_memory_category("state", memory_category)


def _is_unit_interval(value: Any) -> bool:
    """判断 value 是否为 [0, 1] 区间内的实数（bool 不算数值）。"""
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return 0.0 <= float(value) <= 1.0


def verify_memory_item(
    memory: Any,
    expected_kind: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    """校验单条记忆的数据格式。

    Args:
        memory: 待校验的记忆条目。
        expected_kind: 期望的 memory_type（如来自抽取分组 key），为空则不做一致性校验。

    Returns:
        (是否合法, 错误信息列表)。
    """
    errors: List[str] = []

    if not isinstance(memory, dict):
        return False, [f"记忆条目必须是 dict，实际为 {type(memory).__name__}"]

    # 必填字段存在性
    for field in REQUIRED_FIELDS:
        if field not in memory:
            errors.append(f"缺少必填字段 `{field}`")

    # memory_content
    content = memory.get("memory_content")
    if not isinstance(content, str) or not content.strip():
        errors.append("`memory_content` 必须是非空字符串")

    # memory_type
    memory_type = memory.get("memory_type")
    if not isinstance(memory_type, str) or not memory_type.strip():
        errors.append("`memory_type` 必须是非空字符串")
        memory_type = ""
    else:
        memory_type = memory_type.strip().lower()
        if memory_type not in VALID_MEMORY_TYPES:
            errors.append(
                f"`memory_type` 非法: {memory_type!r}，仅允许 {sorted(VALID_MEMORY_TYPES)}"
            )
        elif expected_kind and memory_type != expected_kind:
            errors.append(
                f"`memory_type` 与分组不一致: 期望 {expected_kind!r}，实际 {memory_type!r}"
            )

    # memory_category
    memory_category = memory.get("memory_category")
    if not isinstance(memory_category, str) or not memory_category.strip():
        errors.append("`memory_category` 必须是非空字符串")
    else:
        memory_category = memory_category.strip().lower()
        if not is_valid_memory_category(memory_type, memory_category):
            valid = sorted(VALID_MEMORY_CATEGORIES_BY_TYPE.get(memory_type, set()))
            errors.append(
                f"`memory_category` 非法: type={memory_type!r} category={memory_category!r}，"
                f"该类型合法取值为 {valid}"
            )

    # importance / confidence
    if not _is_unit_interval(memory.get("importance")):
        errors.append("`importance` 必须是 [0, 1] 区间内的数值")
    if not _is_unit_interval(memory.get("confidence")):
        errors.append("`confidence` 必须是 [0, 1] 区间内的数值")

    # cite
    cite = memory.get("cite")
    if not isinstance(cite, list):
        errors.append("`cite` 必须是数组")
    else:
        bad = [c for c in cite if not isinstance(c, int) or isinstance(c, bool)]
        if bad:
            errors.append(f"`cite` 内元素必须均为整数，非法元素: {bad}")

    return (len(errors) == 0), errors


def verify_extracted_memories(
    kind_to_memories: Any,
    strict_kind: bool = True,
) -> Dict[str, Any]:
    """校验 extract_all 返回的整体结构 {kind: [memory, ...]}。

    Args:
        kind_to_memories: 抽取器返回的分组结果。
        strict_kind: 为 True 时校验每条记忆的 memory_type 是否与分组 key 一致。

    Returns:
        {
            "valid": bool,                       # 是否全部合法
            "total": int,                        # 记忆总条数
            "valid_count": int,                  # 合法条数
            "valid_memories": {kind: [memory]},  # 仅保留合法记忆的分组结果
            "errors": [                          # 全部错误明细
                {"kind": str, "index": int, "errors": [str, ...]},
                ...
            ],
        }
    """
    result: Dict[str, Any] = {
        "valid": True,
        "total": 0,
        "valid_count": 0,
        "valid_memories": {},
        "errors": [],
    }

    if not isinstance(kind_to_memories, dict):
        result["valid"] = False
        result["errors"].append(
            {
                "kind": None,
                "index": -1,
                "errors": [f"抽取结果必须是 dict，实际为 {type(kind_to_memories).__name__}"],
            }
        )
        return result

    for kind, memories in kind_to_memories.items():
        valid_list: List[Dict] = []

        if not isinstance(memories, list):
            result["valid"] = False
            result["errors"].append(
                {
                    "kind": kind,
                    "index": -1,
                    "errors": [f"分组 {kind!r} 的值必须是数组，实际为 {type(memories).__name__}"],
                }
            )
            result["valid_memories"][kind] = valid_list
            continue

        expected_kind = kind if strict_kind else None
        for idx, memory in enumerate(memories):
            result["total"] += 1
            ok, errors = verify_memory_item(memory, expected_kind=expected_kind)
            if ok:
                result["valid_count"] += 1
                valid_list.append(memory)
            else:
                result["valid"] = False
                result["errors"].append(
                    {"kind": kind, "index": idx, "errors": errors}
                )

        result["valid_memories"][kind] = valid_list

    if not result["valid"]:
        logger.warning(
            f"[llm_memory_extract_verify] 校验未全部通过: "
            f"{result['valid_count']}/{result['total']} 合法, "
            f"{len(result['errors'])} 处错误"
        )

    return result


__all__ = [
    "REQUIRED_FIELDS",
    "VALID_MEMORY_TYPES",
    "verify_memory_item",
    "verify_extracted_memories",
]


if __name__ == "__main__":
    sample = {
        "profile": [
            {
                "memory_content": "用户在上海工作",
                "memory_type": "profile",
                "memory_category": "fact",
                "importance": 0.8,
                "confidence": 0.9,
                "cite": [0, 2],
            },
            {
                "memory_content": "",  # 非法：空内容
                "memory_type": "profile",
                "memory_category": "unknown_cat",  # 非法：子类不合法
                "importance": 1.5,  # 非法：越界
                "confidence": "high",  # 非法：非数值
                "cite": [0, "x"],  # 非法：含非整数
            },
        ],
        "episodic": [
            {
                "memory_content": "用户计划周末去公园",
                "memory_type": "episodic",
                "memory_category": "life_event",
                "importance": 0.5,
                "confidence": 0.6,
                "cite": [3],
            }
        ],
        "state": [],
    }

    report = verify_extracted_memories(sample)
    import json

    print(json.dumps(report, ensure_ascii=False, indent=2))

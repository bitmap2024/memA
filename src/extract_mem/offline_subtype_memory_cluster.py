#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build per-subtype user memory namespaces from stored memories.

The script reads one user's memories from Qdrant, groups them by subtype
(`memory_category` by default), asks the configured LLM to summarize each group,
and writes one Markdown namespace per subtype.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from loguru import logger
from qdrant_client.http import models


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
SRC_ROOT = CURRENT_FILE.parents[1]
for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from api.llm_api import LLMApi  # noqa: E402
from api.qdrant_api import QdrantMemoryClient  # noqa: E402
from config.config import Config  # noqa: E402
from config.memory_item import MemoryItem  # noqa: E402


SUBTYPE_SUMMARY_SYSTEM_PROMPT = """你是用户记忆 namespace 构建模块。
你会收到同一个用户、同一个 subtype 下的多条记忆。请把它们整理成可长期复用的 Markdown 用户记忆 namespace。

要求：
- 只保留与用户相关、对后续个性化回应有帮助的信息。
- 合并重复、近似、可归纳的信息；如果信息冲突，优先采用更新时间更新或更具体的表述。
- 不要编造输入中没有的信息。
- 使用中文输出。
- 输出 Markdown，不要输出 JSON，不要解释你的处理过程。"""


SUBTYPE_SUMMARY_USER_PROMPT = """用户 ID：{user_id}
Subtype：{subtype}
记忆条数：{memory_count}

请生成这个 subtype 的用户记忆 namespace。建议结构：

# {subtype}

## 核心概览
- ...

## 详细记忆
- ...

## 后续互动提示
- ...

以下是待整理的原始记忆，按时间排序：

{memories}
"""


class OfflineSubtypeMemoryCluster:
    """Group user memories by subtype and summarize each group into Markdown."""

    def __init__(
        self,
        config: Config = Config,
        vector_store: Optional[QdrantMemoryClient] = None,
        llm: Optional[LLMApi] = None,
    ) -> None:
        self.config = config
        self.vector_store = vector_store or QdrantMemoryClient(
            collection_name=config.qdrant.COLLECTION,
            vector_size=config.qdrant.VECTOR_SIZE,
            distance=config.qdrant.DISTANCE,
        )
        self.llm = llm or LLMApi.from_params(
            api_key=config.llm.API_KEY,
            base_url=config.llm.BASE_URL,
            model=config.llm.MODEL,
            temperature=0.2,
            max_tokens=max(2048, getattr(config.llm, "MAX_TOKENS", 2048)),
            top_p=getattr(config.llm, "TOP_P", 0.95),
            timeout=getattr(config.llm, "TIMEOUT", 60),
        )

    def load_user_memories(
        self,
        user_id: str,
        user_field: str = "child_id",
        agent_id: Optional[str] = None,
        scroll_size: int = 512,
    ) -> List[MemoryItem]:
        """Read all memories for one user from Qdrant."""
        payload_filter = self._build_user_filter(
            user_id=user_id,
            user_field=user_field,
            agent_id=agent_id,
        )
        records = self.vector_store.scroll_all(
            payload_filter=payload_filter,
            limit=scroll_size,
            with_vectors=False,
        )
        memories = [self._record_to_memory_item(record, user_id=user_id) for record in records]
        memories = [memory for memory in memories if memory.memory_content.strip()]
        return sorted(memories, key=lambda item: (item.updated_at or item.created_at, item.id))

    def group_by_subtype(
        self,
        memories: Iterable[MemoryItem],
        subtype_field: str = "memory_category",
    ) -> Dict[str, List[MemoryItem]]:
        """Group memories by subtype, falling back to compatible field names."""
        groups: Dict[str, List[MemoryItem]] = {}
        for memory in memories:
            memory_dict = asdict(memory)
            subtype = (
                memory_dict.get(subtype_field)
                or memory_dict.get("memory_category")
                or memory_dict.get("subtype")
                or memory_dict.get("memory_subtype")
                or memory_dict.get("memory_type")
                or "uncategorized"
            )
            subtype = str(subtype).strip() or "uncategorized"
            groups.setdefault(subtype, []).append(memory)
        return dict(sorted(groups.items(), key=lambda item: item[0]))

    def build_namespaces(
        self,
        user_id: str,
        output_dir: Path,
        user_field: str = "child_id",
        agent_id: Optional[str] = None,
        subtype_field: str = "memory_category",
        scroll_size: int = 512,
        max_chars_per_call: int = 12000,
    ) -> Dict[str, Path]:
        """Create one Markdown namespace file for each subtype."""
        memories = self.load_user_memories(
            user_id=user_id,
            user_field=user_field,
            agent_id=agent_id,
            scroll_size=scroll_size,
        )
        logger.info(f"读取到 {len(memories)} 条用户记忆: user_id={user_id}")

        groups = self.group_by_subtype(memories, subtype_field=subtype_field)
        if not groups:
            logger.warning("没有可写入 namespace 的记忆")
            return {}

        user_output_dir = output_dir / self._safe_filename(user_id)
        user_output_dir.mkdir(parents=True, exist_ok=True)

        written: Dict[str, Path] = {}
        for subtype, subtype_memories in groups.items():
            markdown = self.summarize_subtype(
                user_id=user_id,
                subtype=subtype,
                memories=subtype_memories,
                max_chars_per_call=max_chars_per_call,
            )
            target_path = user_output_dir / f"{self._safe_filename(subtype)}.md"
            target_path.write_text(markdown.strip() + "\n", encoding="utf-8")
            written[subtype] = target_path
            logger.info(f"写入 subtype namespace: {subtype} -> {target_path}")

        return written

    def summarize_subtype(
        self,
        user_id: str,
        subtype: str,
        memories: List[MemoryItem],
        max_chars_per_call: int = 12000,
    ) -> str:
        """Summarize one subtype. Large groups are summarized in chunks first."""
        chunks = self._chunk_memories(memories, max_chars_per_call=max_chars_per_call)
        if len(chunks) == 1:
            return self._call_summary_llm(user_id=user_id, subtype=subtype, memories=chunks[0])

        partial_summaries: List[MemoryItem] = []
        for index, chunk in enumerate(chunks, start=1):
            chunk_subtype = f"{subtype} / chunk {index}"
            summary = self._call_summary_llm(
                user_id=user_id,
                subtype=chunk_subtype,
                memories=chunk,
            )
            partial_summaries.append(
                MemoryItem(
                    id=f"{subtype}_chunk_{index}",
                    user_id=user_id,
                    memory_content=summary,
                    memory_type="namespace_summary",
                    memory_category=subtype,
                )
            )

        return self._call_summary_llm(
            user_id=user_id,
            subtype=subtype,
            memories=partial_summaries,
        )

    def _call_summary_llm(
        self,
        user_id: str,
        subtype: str,
        memories: List[MemoryItem],
    ) -> str:
        memory_text = self._format_memories(memories)
        prompt = SUBTYPE_SUMMARY_USER_PROMPT.format(
            user_id=user_id,
            subtype=subtype,
            memory_count=len(memories),
            memories=memory_text,
        )
        return self.llm.chat(
            messages=[
                {"role": "system", "content": SUBTYPE_SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )

    def _build_user_filter(
        self,
        user_id: str,
        user_field: str,
        agent_id: Optional[str],
    ) -> models.Filter:
        extra_must: List[models.Condition] = []
        child_id: Optional[str] = None

        if user_field == "child_id":
            child_id = user_id
        else:
            extra_must.append(
                models.FieldCondition(
                    key=user_field,
                    match=models.MatchValue(value=user_id),
                )
            )

        payload_filter = QdrantMemoryClient.build_filter(
            child_id=child_id,
            agent_id=agent_id,
            extra_must=extra_must,
        )
        if payload_filter is None:
            raise ValueError("user_id 不能为空")
        return payload_filter

    @staticmethod
    def _record_to_memory_item(record: Dict[str, Any], user_id: str) -> MemoryItem:
        return MemoryItem(
            id=str(record.get("memory_id") or record.get("_id") or ""),
            user_id=str(record.get("user_id") or record.get("child_id") or user_id),
            memory_content=str(record.get("memory_content") or record.get("content") or ""),
            memory_type=str(record.get("memory_type") or ""),
            memory_category=str(
                record.get("memory_category")
                or record.get("subtype")
                or record.get("memory_subtype")
                or ""
            ),
            created_at=str(record.get("created_at") or ""),
            updated_at=str(record.get("updated_at") or ""),
            emedding=list(record.get("vector") or []),
            retrieval_count=int(record.get("retrieval_count") or 0),
            last_retrieved_at=str(record.get("last_retrieved_at") or ""),
            source=str(record.get("source") or record.get("session_id") or ""),
        )

    @staticmethod
    def _format_memories(memories: List[MemoryItem]) -> str:
        lines: List[str] = []
        for index, memory in enumerate(memories, start=1):
            timestamp = memory.updated_at or memory.created_at or "unknown_time"
            source = f", source={memory.source}" if memory.source else ""
            memory_type = f", type={memory.memory_type}" if memory.memory_type else ""
            lines.append(
                f"{index}. [{timestamp}{memory_type}{source}] {memory.memory_content.strip()}"
            )
        return "\n".join(lines)

    @staticmethod
    def _chunk_memories(
        memories: List[MemoryItem],
        max_chars_per_call: int,
    ) -> List[List[MemoryItem]]:
        chunks: List[List[MemoryItem]] = []
        current: List[MemoryItem] = []
        current_chars = 0

        for memory in memories:
            formatted_len = len(memory.memory_content) + len(memory.updated_at) + 64
            if current and current_chars + formatted_len > max_chars_per_call:
                chunks.append(current)
                current = []
                current_chars = 0
            current.append(memory)
            current_chars += formatted_len

        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _safe_filename(value: str) -> str:
        normalized = re.sub(r"[^\w\-.]+", "_", value.strip(), flags=re.UNICODE)
        normalized = normalized.strip("._")
        return normalized or "uncategorized"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按 subtype 汇总用户记忆，并生成 Markdown namespace 文件。"
    )
    parser.add_argument("--user-id", required=True, help="用户 ID")
    parser.add_argument(
        "--user-field",
        default="child_id",
        help="Qdrant payload 中表示用户 ID 的字段，默认 child_id；如果使用 MemoryItem，可传 user_id。",
    )
    parser.add_argument("--agent-id", default=None, help="可选，按 agent_id 进一步过滤")
    parser.add_argument(
        "--subtype-field",
        default="memory_category",
        help="作为 subtype 的字段，默认 memory_category。",
    )
    parser.add_argument(
        "--output-dir",
        default="memory_namespaces",
        help="namespace Markdown 输出目录，默认 memory_namespaces。",
    )
    parser.add_argument("--scroll-size", type=int, default=512, help="Qdrant scroll 批大小")
    parser.add_argument(
        "--max-chars-per-call",
        type=int,
        default=12000,
        help="单次 LLM 调用包含的最大原始记忆字符数。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cluster = OfflineSubtypeMemoryCluster(Config)
    written = cluster.build_namespaces(
        user_id=args.user_id,
        user_field=args.user_field,
        agent_id=args.agent_id,
        subtype_field=args.subtype_field,
        output_dir=Path(args.output_dir),
        scroll_size=args.scroll_size,
        max_chars_per_call=args.max_chars_per_call,
    )
    logger.info(f"完成，共写入 {len(written)} 个 subtype namespace")


if __name__ == "__main__":
    main()

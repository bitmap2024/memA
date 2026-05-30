#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CategoryDocBuilder：按 (memory_type, memory_category) 聚合用户活跃记忆，
调用 LLM 生成 category.md 并写入 OSS（或本地兜底）。"""

from __future__ import annotations

import concurrent.futures
import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
from config.releatiion_schema import MemoryItem
from prompt.memory_prompt.chat_bot.prompt_zh import CATEGORY_DOC_PROMPT
from config.setting import Config
from src.db import MemoryStoreClient, create_memory_store
from src.llm.openai_llm import LLMApi
from src.utils.oss_manage import OssDocumentStore


class CategoryDocBuilder:
    """把活跃记忆整理成 Markdown，每个 (memory_type, category) 一个文件，写入 OSS。"""

    def __init__(
        self,
        store: Optional[MemoryStoreClient] = None,
        oss: Optional[OssDocumentStore] = None,
        llm: Optional[LLMApi] = None,
    ) -> None:
        self.store = store or create_memory_store()
        self.oss = oss or OssDocumentStore()
        self.llm = llm or LLMApi.from_params(
            api_key=Config.llm.API_KEY,
            base_url=Config.llm.BASE_URL,
            model=Config.llm.MODEL,
            temperature=0.2,
            max_tokens=Config.llm.MAX_TOKENS,
            top_p=Config.llm.TOP_P,
            timeout=Config.llm.TIMEOUT,
        )

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------
    def build_for_user(
        self,
        user_id: str,
        max_workers: int = 4,
    ) -> Dict[str, Any]:
        memories = self.store.list_user_memories(user_id=user_id, status="active")
        if not memories:
            logger.info(f"[CategoryDocBuilder] 用户 {user_id} 无活跃记忆，跳过")
            return {"user_id": user_id, "categories": 0, "written": []}

        grouped: Dict[Tuple[str, str], List[MemoryItem]] = defaultdict(list)
        for mem in memories:
            grouped[(mem.memory_type, mem.memory_category)].append(mem)

        report: Dict[str, Any] = {"user_id": user_id, "categories": len(grouped), "written": []}

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    self._build_one_category,
                    user_id=user_id,
                    memory_type=mt,
                    memory_category=mc,
                    memories=mems,
                ): (mt, mc)
                for (mt, mc), mems in grouped.items()
            }
            for fut in concurrent.futures.as_completed(futures):
                (mt, mc) = futures[fut]
                try:
                    key = fut.result()
                    if key:
                        report["written"].append({"memory_type": mt, "memory_category": mc, "key": key})
                except Exception as e:
                    logger.warning(f"[CategoryDocBuilder] 生成 {mt}/{mc} 失败: {e}")

        logger.info(
            f"[CategoryDocBuilder] 用户 {user_id} category.md 完成: {len(report['written'])}/{report['categories']}"
        )
        return report

    # ------------------------------------------------------------------
    # 单个 category 文档
    # ------------------------------------------------------------------
    def _build_one_category(
        self,
        user_id: str,
        memory_type: str,
        memory_category: str,
        memories: List[MemoryItem],
    ) -> Optional[str]:
        markdown = self._call_llm(user_id, memory_type, memory_category, memories)
        if not markdown:
            # LLM 兜底：用模板渲染
            sorted_mems = sorted(memories, key=lambda m: (m.importance, m.created_at), reverse=True)
            bullets = [
                f"({m.importance:.2f}) {m.memory_content}" for m in sorted_mems
            ]
            summary = f"{user_id} 在 {memory_type}/{memory_category} 下共有 {len(memories)} 条活跃记忆。"
            markdown = self.oss.render_memory_section(
                memory_type=memory_type,
                memory_category=memory_category,
                summary=summary,
                bullets=bullets,
            )
        else:
            markdown = self._normalize_llm_markdown(markdown, memory_type, memory_category)

        return self.oss.put(
            user_id=user_id,
            memory_type=memory_type,
            memory_category=memory_category,
            content=markdown,
        )

    def _call_llm(
        self,
        user_id: str,
        memory_type: str,
        memory_category: str,
        memories: List[MemoryItem],
    ) -> Optional[str]:
        sorted_mems = sorted(memories, key=lambda m: (m.importance, m.created_at), reverse=True)
        payload = [
            {
                "id": m.memory_id,
                "memory_content": m.memory_content,
                "importance": m.importance,
                "confidence": m.confidence,
                "created_at": m.created_at,
            }
            for m in sorted_mems[:60]
        ]
        prompt = CATEGORY_DOC_PROMPT.format(
            user_id=user_id,
            memory_type=memory_type,
            memory_category=memory_category,
            memories_json=json.dumps(payload, ensure_ascii=False, indent=2),
        )
        try:
            return self.llm.chat(messages=[{"role": "user", "content": prompt}])
        except Exception as e:
            logger.warning(f"[CategoryDocBuilder] LLM 生成 markdown 失败: {e}")
            return None

    @staticmethod
    def _normalize_llm_markdown(markdown: str, memory_type: str, memory_category: str) -> str:
        text = markdown.strip()
        # 去掉 ```markdown ``` 包裹
        fence = re.match(r"```(?:markdown)?\s*(.*?)\s*```", text, flags=re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        header = f"# {memory_type.capitalize()} Memory / {memory_category}\n"
        if text.startswith("#"):
            return text + "\n"
        return header + "\n" + text + "\n"

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""OSS 文档存储：按 user_id 组织 Chat Bot Memory 目录结构。

目录结构（README 中定义）：
    <prefix>/<user_id>/Profile Memory/fact.md
    <prefix>/<user_id>/Profile Memory/preference.md
    <prefix>/<user_id>/Episodic Memory/conversation_event.md
    ...
    <prefix>/<user_id>/Core Memory/identity.md

实现：
- 优先调用阿里云 OSS（如果 oss2 已安装且 ak/secret 不为占位符）；
- 否则写入本地兜底目录（config.oss.LOCAL_FALLBACK_DIR），便于本地 mock。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

from loguru import logger

from config.setting import Config, OssConfig


try:  # pragma: no cover
    import oss2  # type: ignore

    _HAS_OSS2 = True
except Exception:  # pragma: no cover
    oss2 = None
    _HAS_OSS2 = False


PROFILE_CATEGORIES = (
    "fact",
    "preference",
    "relationship",
    "goal",
    "constraint",
    "skill",
    "portrait",
    "communication_style",
)
EPISODIC_CATEGORIES = (
    "conversation_event",
    "life_event",
    "project_event",
    "relationship_event",
    "decision_event",
    "unresolved_event",
)
STATE_CATEGORIES = (
    "current_focus",
    "recent_mood",
    "emotional_need",
    "relationship_state",
    "task_state",
    "short_term_context",
)
CORE_BLOCKS = (
    "identity",
    "interaction_guide",
    "current_focus",
    "emotional_state",
    "pending_followups",
)

TYPE_TO_DIR = {
    "profile": "Profile Memory",
    "episodic": "Episodic Memory",
    "state": "State Memory",
    "core": "Core Memory",
}

CATEGORY_TO_TYPE = {
    **{c: "profile" for c in PROFILE_CATEGORIES},
    **{c: "episodic" for c in EPISODIC_CATEGORIES},
    **{c: "state" for c in STATE_CATEGORIES},
    **{c: "core" for c in CORE_BLOCKS},
}


class OssDocumentStore:
    """以 user_id 为命名空间的记忆文档存储。"""

    def __init__(self, cfg: Optional[OssConfig] = None):
        self.cfg = cfg or Config.oss
        self._bucket = None
        self._use_local = True

        if (
            _HAS_OSS2
            and self.cfg.ACCESS_KEY_ID
            and self.cfg.ACCESS_KEY_SECRET
            and not self.cfg.ACCESS_KEY_ID.startswith("LTAI5tJ6")  # 默认占位 ak
        ):
            try:  # pragma: no cover
                auth = oss2.Auth(self.cfg.ACCESS_KEY_ID, self.cfg.ACCESS_KEY_SECRET)
                self._bucket = oss2.Bucket(auth, self.cfg.ENDPOINT, self.cfg.BUCKET)
                self._bucket.get_bucket_info()
                self._use_local = False
                logger.info(
                    f"[OssDocumentStore] 启用 OSS 后端: {self.cfg.BUCKET}@{self.cfg.ENDPOINT}"
                )
            except Exception as e:  # pragma: no cover
                logger.warning(f"[OssDocumentStore] OSS 不可用，使用本地兜底: {e}")
                self._use_local = True

        if self._use_local:
            Path(self.cfg.LOCAL_FALLBACK_DIR).mkdir(parents=True, exist_ok=True)
            logger.info(
                f"[OssDocumentStore] 使用本地兜底目录: {self.cfg.LOCAL_FALLBACK_DIR}"
            )

    # ------------------------------------------------------------------
    # 路径计算
    # ------------------------------------------------------------------
    def object_key(
        self,
        user_id: str,
        memory_type: str,
        memory_category: str,
    ) -> str:
        memory_type = memory_type or CATEGORY_TO_TYPE.get(memory_category, "profile")
        dir_name = TYPE_TO_DIR.get(memory_type, "Profile Memory")
        category = memory_category or "uncategorized"
        return f"{self.cfg.PREFIX}/{self._safe(user_id)}/{dir_name}/{self._safe(category)}.md"

    def _safe(self, value: str) -> str:
        import re

        return re.sub(r"[^\w\-.]+", "_", (value or "").strip(), flags=re.UNICODE) or "unknown"

    # ------------------------------------------------------------------
    # 读写接口
    # ------------------------------------------------------------------
    def put(self, user_id: str, memory_type: str, memory_category: str, content: str) -> str:
        key = self.object_key(user_id, memory_type, memory_category)
        if self._use_local:
            target = Path(self.cfg.LOCAL_FALLBACK_DIR) / key
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return str(target)

        # pragma: no cover - 真实 OSS
        self._bucket.put_object(key, content.encode("utf-8"))
        return key

    def get(self, user_id: str, memory_type: str, memory_category: str) -> Optional[str]:
        key = self.object_key(user_id, memory_type, memory_category)
        if self._use_local:
            target = Path(self.cfg.LOCAL_FALLBACK_DIR) / key
            if not target.exists():
                return None
            return target.read_text(encoding="utf-8")

        try:  # pragma: no cover
            stream = self._bucket.get_object(key)
            return stream.read().decode("utf-8")
        except Exception:
            return None

    def list_user_categories(self, user_id: str) -> List[Dict[str, str]]:
        results: List[Dict[str, str]] = []
        if self._use_local:
            user_root = Path(self.cfg.LOCAL_FALLBACK_DIR) / self.cfg.PREFIX / self._safe(user_id)
            if not user_root.exists():
                return results
            for type_dir in user_root.iterdir():
                if not type_dir.is_dir():
                    continue
                for md in type_dir.glob("*.md"):
                    results.append(
                        {
                            "type_dir": type_dir.name,
                            "category": md.stem,
                            "key": str(md.relative_to(Path(self.cfg.LOCAL_FALLBACK_DIR))),
                        }
                    )
            return results

        # pragma: no cover
        prefix = f"{self.cfg.PREFIX}/{self._safe(user_id)}/"
        for obj in oss2.ObjectIterator(self._bucket, prefix=prefix):
            parts = obj.key.replace(prefix, "").split("/")
            if len(parts) == 2 and parts[1].endswith(".md"):
                results.append(
                    {
                        "type_dir": parts[0],
                        "category": parts[1].replace(".md", ""),
                        "key": obj.key,
                    }
                )
        return results

    def render_memory_section(
        self,
        memory_type: str,
        memory_category: str,
        summary: str,
        bullets: Iterable[str],
    ) -> str:
        """渲染单个 category.md 的标准 Markdown 模板。"""
        type_dir = TYPE_TO_DIR.get(memory_type, "Memory")
        bullet_lines = "\n".join([f"- {b}" for b in bullets if (b or "").strip()])
        return (
            f"# {type_dir} / {memory_category}\n\n"
            f"## 摘要\n{summary.strip()}\n\n"
            f"## 详细记忆\n{bullet_lines or '- (暂无)'}\n"
        )

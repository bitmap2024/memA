"""离线 update / merge / category 文档生成 prompts。"""

from __future__ import annotations


MERGE_MEMORY_PROMPT = """# 记忆合并模块

你是 AI 女友的【记忆合并模块】。

我会给你**同一个用户、同一个 memory_type + memory_category**下的一组相似记忆。
你的任务是：判断是否需要合并，并在合并时生成一条新的权威 (canonical) 记忆。

## 输入

memory_type: {memory_type}
memory_category: {memory_category}

候选记忆（按 updated_at 倒序，第一条最新）：
{memories_json}

## 判断与产出规则

1. 如果这组记忆中存在**含义冲突**（例如对同一个事实给出不一致结论），优先保留时间更新、用户明确表达的版本，并在 `merged_content` 中明确写出最新结论。
2. 如果这组记忆都在表达**同一件事的不同表述**，请合并为信息量最大、最完整的一条。
3. 如果它们其实**不是同一件事**（被错误聚到一起），返回 `should_merge=false`，不要强行合并。
4. 合并后的 `importance` 不低于源记忆的最大值；`confidence` 取源记忆的中位数。
5. 不允许编造源记忆中没有出现过的信息。

## 输出格式

只输出合法 JSON：

{{
  "should_merge": true,
  "merged_content": "合并后的记忆陈述句（中文）",
  "memory_type": "{memory_type}",
  "memory_category": "{memory_category}",
  "importance": 0.0,
  "confidence": 0.0,
  "archived_reason": "为什么旧记忆可以归档（一句话）"
}}

如果判断不合并，输出：

{{"should_merge": false, "reason": "为什么不合并（一句话）"}}
"""


UPDATE_MEMORY_PROMPT = """# 记忆修订模块

你是 AI 女友的【记忆修订模块】。

target_memory（当前的权威记忆）：
{target_memory}

新的候选证据（来自更早或更新的会话）：
{candidate_memories_json}

请基于证据决定 target_memory 的下一步动作：

- `keep`：target_memory 仍然有效，不修改。
- `update`：根据证据，对 target_memory 做最小幅度修订。
- `archive`：target_memory 已经过时或被证据否定，应当归档。

## 输出格式

只输出合法 JSON：

{{
  "action": "keep | update | archive",
  "new_content": "如果 action=update，则给出新的记忆陈述句；否则为空字符串",
  "importance": 0.0,
  "confidence": 0.0,
  "reason": "一句话理由"
}}
"""


CATEGORY_DOC_PROMPT = """# 用户记忆 Category 整理模块

你是 AI 女友的【记忆 Category 整理模块】。

请把以下属于同一个用户、同一个 (memory_type, memory_category) 下的活跃记忆，
整理成一份**可长期复用的 Markdown 文档**，写入用户 OSS 知识库。

输入：

user_id: {user_id}
memory_type: {memory_type}
memory_category: {memory_category}
活跃记忆列表：
{memories_json}

输出要求：

- 顶部一段不超过 60 字的「摘要」描述当前这个 category 下用户的核心画像。
- 之后用要点列表整理具体记忆，按重要度从高到低。
- 不要罗列重复信息，不要编造未提供的信息。
- 直接输出 Markdown，不要解释。

输出格式：

```markdown
## 摘要
<60 字以内>

## 详细记忆
- ...
- ...
```
"""


__all__ = [
    "MERGE_MEMORY_PROMPT",
    "UPDATE_MEMORY_PROMPT",
    "CATEGORY_DOC_PROMPT",
]

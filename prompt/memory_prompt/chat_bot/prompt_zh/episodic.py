"""Episodic memory extraction prompt for the chat bot."""

MEMORY_KIND = "episodic"

EPISODIC_ROLE_MARKDOWN = """# Episodic Memory 抽取

你是ai产品的【Episodic Memory 抽取模块】。

你的任务是从用户与AI的聊天中，抽取有明确发生背景、时间线意义或关系推进价值的事件记忆。

Episodic Memory 记录“发生了什么”，不是稳定画像，也不是当前状态。

记忆是稀缺资源：宁可漏记，也不要错记、滥记或记录无意义闲聊。
"""

EPISODIC_SCHEMA_MARKDOWN = """## Memory Schema

固定输出 `memory_type = "episodic"`，并且只使用以下 `memory_category`：

| memory_category | 含义 | 示例 |
| --- | --- | --- |
| `conversation_event` | 对未来互动有复用价值的重要对话事件。 | 用户第一次告诉 AI 自己希望被叫“哥哥”。 |
| `life_event` | 用户生活中已经发生或即将发生的重要事件。 | 用户下周要搬到上海；用户昨天通过了驾照考试。 |
| `project_event` | 用户工作、学习、创作、健身等长期项目中的阶段性事件。 | 用户完成了论文初稿；用户开始准备产品经理面试。 |
| `relationship_event` | 用户与伴侣、家人、朋友、宠物、同事等关系中的重要事件。 | 用户和朋友因为误会吵架；用户给妈妈准备生日礼物。 |
| `decision_event` | 用户做出的重要选择、承诺、取舍或改变方向。 | 用户决定暂时不跳槽；用户决定减少熬夜。 |
| `unresolved_event` | 仍未结束、需要后续关心或跟进的事件。 | 用户还没决定是否参加聚会；用户正在等体检结果。 |"""

EPISODIC_SCOPE_MARKDOWN = """## 只记录这些事件

只有同时满足以下条件的事件才写入 Episodic Memory：

- 事件和用户有关，或会影响用户与 AI 的后续互动。
- 事件有明确触发、过程、结果、决定或待跟进点。
- 事件对未来问候、关心、回顾、陪伴或关系延续有帮助。

禁止记录：

- 没有后续价值的日常流水账。
- 单句情绪宣泄，除非包含明确事件原因。
- AI 女友自己的行为或承诺，除非它是用户明确在意的互动节点。
- 无法判断时间、主体或具体发生内容的事件。
- 色情露骨细节、违法行为指导、敏感身份推断等不应进入记忆库的内容。"""

EPISODIC_EXTRACTION_RULES_MARKDOWN = """## 抽取规则

- 每条 `content` 必须说明事件主体、事件内容，以及必要的时间或上下文。
- 如果事件还未解决，优先归为 `unresolved_event`，并在 `content` 中保留需要跟进的点。
- 如果对话中出现修正或更新，只保留最新事件状态。
- 不要把同一个事件拆成多条重复记忆。
- 不要把长期稳定偏好、性格、能力误写为 Episodic Memory；这些属于 Profile Memory。
- 每条记忆必须在 `cite` 字段中标注来源消息索引，对应待处理对话中 `[n]` 的编号。一条记忆可引用多条消息。
- 不要输出解释、推理过程或 Markdown。"""

EPISODIC_OUTPUT_FORMAT_MARKDOWN = """## 输出格式

只输出合法 JSON。

当存在符合要求的记忆时，额外用一句话概括本段对话的主题摘要，写入 `topic`：

{{
  "topic": "对本段对话的简要主题摘要（一句话）",
  "memories": [
    {{
      "memory_type": "episodic",
      "memory_category": "conversation_event | life_event | project_event | relationship_event | decision_event | unresolved_event",
      "content": "完整、清晰、可用于未来回顾或跟进的事件陈述句",
      "cite": [0, 2],
      "importance": 0.0,
      "confidence": 0.0
    }}
  ]
}}

字段说明：
- `topic`：仅在 `memories` 非空时输出，对本段对话进行一句话主题概括，便于后续检索与定位。`memories` 为空时禁止输出该字段。
- `cite`：整数数组，对应待处理对话中 `[n]` 的编号，表示该记忆的来源消息。
- `importance`：0~1，对未来互动/关系延续越重要越接近 1。`unresolved_event` 默认不低于 0.6。
- `confidence`：0~1，事件主体、时间、内容越清晰越接近 1。

如果没有值得记录的内容，则不要输出 `topic`，直接输出：

{{"memories": []}}"""

INPUT_MARKDOWN = """## 待处理对话

{text}"""

EPISODIC_EXTRACT_MEMORY_PROMPT_MARKDOWN = "\n\n".join(
    [
        EPISODIC_ROLE_MARKDOWN,
        EPISODIC_SCHEMA_MARKDOWN,
        EPISODIC_SCOPE_MARKDOWN,
        EPISODIC_EXTRACTION_RULES_MARKDOWN,
        EPISODIC_OUTPUT_FORMAT_MARKDOWN,
        INPUT_MARKDOWN,
    ]
)

EXTRACT_MEMORY_PROMPT = EPISODIC_EXTRACT_MEMORY_PROMPT_MARKDOWN

__all__ = [
    "MEMORY_KIND",
    "EPISODIC_ROLE_MARKDOWN",
    "EPISODIC_SCHEMA_MARKDOWN",
    "EPISODIC_SCOPE_MARKDOWN",
    "EPISODIC_EXTRACTION_RULES_MARKDOWN",
    "EPISODIC_OUTPUT_FORMAT_MARKDOWN",
    "INPUT_MARKDOWN",
    "EPISODIC_EXTRACT_MEMORY_PROMPT_MARKDOWN",
    "EXTRACT_MEMORY_PROMPT",
]
if __name__ == "__main__":
    print(EPISODIC_EXTRACT_MEMORY_PROMPT_MARKDOWN)
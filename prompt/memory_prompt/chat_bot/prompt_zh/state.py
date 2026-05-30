"""State memory extraction prompt for the chat bot."""

MEMORY_KIND = "state"

STATE_ROLE_MARKDOWN = """# State Memory 抽取

你是 AI产品的【State Memory 抽取模块】。

你的任务是从用户与 AI的聊天中，抽取短期有效、需要近期响应或持续跟进的当前状态。

State Memory 记录“用户现在处在什么状态”，不是长期画像，也不是完整历史事件。

记忆是稀缺资源：宁可漏记，也不要错记、滥记或记录无意义闲聊。
"""

STATE_SCHEMA_MARKDOWN = """## Memory Schema

固定输出 `memory_type = "state"`，并且只使用以下 `memory_category`：

| memory_category | 含义 | 示例 |
| --- | --- | --- |
| `current_focus` | 用户当前最关注、正在处理或反复提到的主题。 | 用户最近主要在准备产品经理面试。 |
| `recent_mood` | 用户近期明确表达的心情，情绪状态。 | 用户今晚因为加班感到疲惫和委屈。 |
| `emotional_need` | 用户当下需要的陪伴、安抚、鼓励、空间或互动方式。 | 用户现在更需要被安慰，而不是被建议。 |
| `relationship_state` | 用户近期关系中的状态、 tension、亲密进展或需要注意的变化。 | 用户和朋友的误会还没有解开。 |
| `task_state` | 用户当前任务、计划、待办的进展状态。 | 用户明天要提交论文初稿，目前还差结论部分。 |
| `short_term_context` | 对近期连续对话有帮助的临时上下文。 | 用户今晚在高铁上，回复可能比较慢。 |"""

STATE_SCOPE_MARKDOWN = """## 只记录这些状态

只有同时满足以下条件的信息才写入 State Memory：

- 信息反映用户当前或近期状态。
- 信息对接下来几轮或近期几天的回应、关心、提醒、跟进有帮助。
- 信息不是长期稳定事实；如果长期稳定，应交给 Profile Memory。
- 信息不是完整事件回顾；如果重点是事件本身，应交给 Episodic Memory。

禁止记录：

- 没有可跟进价值的普通闲聊。
- 没有明确触发或内容的泛泛情绪词。
- 已经结束且无需再关心的临时状态。
- AI 自己的状态、承诺或情绪。
- 无法确定主体、时间或具体状态的信息。
- 色情露骨细节、违法行为指导、敏感身份推断等不                                                                                 应进入记忆库的内容。"""

STATE_EXTRACTION_RULES_MARKDOWN = """## 抽取规则

- 每条 `content` 必须说明状态主体、当前状态，以及必要的短期上下文。
- 如果状态有明确期限、时间点或待跟进节点，应写入 `content`。
- 如果后文显示状态已经变化，只保留最新状态。
- 不要把同一状态拆成多条重复记忆。
- 不要把长期偏好、性格、能力误写为 State Memory。
- 每条记忆必须在 `cite` 字段中标注来源消息索引，对应待处理对话中 `[n]` 的编号。一条记忆可引用多条消息。
- 不要输出解释、推理过程或 Markdown。"""

STATE_OUTPUT_FORMAT_MARKDOWN = """## 输出格式

只输出合法 JSON。

当存在符合要求的记忆时，额外用一句话概括本段对话的主题摘要，写入 `topic`：

{{
  "topic": "对本段对话的简要主题摘要（一句话）",
  "memories": [
    {{
      "memory_type": "state",
      "memory_category": "current_focus | recent_mood | emotional_need | relationship_state | task_state | short_term_context",
      "content": "完整、清晰、近期可用于回应或跟进的状态陈述句",
      "cite": [0, 2],
      "importance": 0.0,
      "confidence": 0.0
    }}
  ]
}}

字段说明：
- `topic`：仅在 `memories` 非空时输出，对本段对话进行一句话主题概括，便于后续检索与定位。`memories` 为空时禁止输出该字段。
- `cite`：整数数组，对应待处理对话中 `[n]` 的编号，表示该记忆的来源消息。
- `importance`：0~1，State Memory 偏短期，>=0.7 通常代表近 1-7 天需要主动跟进的状态。
- `confidence`：0~1，对照对话原文越明确越接近 1。

如果没有值得记录的内容，则不要输出 `topic`，直接输出：

{{"memories": []}}"""

INPUT_MARKDOWN = """## 待处理对话

{text}"""

STATE_EXTRACT_MEMORY_PROMPT_MARKDOWN = "\n\n".join(
    [
        STATE_ROLE_MARKDOWN,
        STATE_SCHEMA_MARKDOWN,
        STATE_SCOPE_MARKDOWN,
        STATE_EXTRACTION_RULES_MARKDOWN,
        STATE_OUTPUT_FORMAT_MARKDOWN,
        INPUT_MARKDOWN,
    ]
)

EXTRACT_MEMORY_PROMPT = STATE_EXTRACT_MEMORY_PROMPT_MARKDOWN

__all__ = [
    "MEMORY_KIND",
    "STATE_ROLE_MARKDOWN",
    "STATE_SCHEMA_MARKDOWN",
    "STATE_SCOPE_MARKDOWN",
    "STATE_EXTRACTION_RULES_MARKDOWN",
    "STATE_OUTPUT_FORMAT_MARKDOWN",
    "INPUT_MARKDOWN",
    "STATE_EXTRACT_MEMORY_PROMPT_MARKDOWN",
    "EXTRACT_MEMORY_PROMPT",
]

"""Profile memory extraction prompt for the chat bot."""

MEMORY_KIND = "profile"

PROFILE_ROLE_MARKDOWN = """# Profile Memory 抽取

你是 AI 女友产品的【Profile Memory 抽取模块】。

你的任务是从用户与 AI 女友的聊天中，抽取稳定、长期可复用、能帮助未来亲密陪伴和个性化回应的用户画像记忆。

记忆是稀缺资源：宁可漏记，也不要错记、滥记或记录无意义闲聊。"""

PROFILE_SCHEMA_MARKDOWN = """## Memory Schema

固定输出 `memory_type = "profile"`，并且只使用以下 `memory_category`：

| memory_category | 含义 | 示例 |
| --- | --- | --- |
| `fact` | 用户长期稳定的客观事实，包括身份、生活状态、工作/学习、居住城市、重要经历等。 | 用户在上海工作；用户养了一只叫奶茶的猫。 |
| `preference` | 用户明确、可复现的喜欢、讨厌、习惯、称呼偏好、互动边界。 | 用户喜欢被叫“哥哥”；用户不喜欢被频繁追问隐私。 |
| `relationship` | 用户提到的具体重要关系，包括伴侣、家人、朋友、宠物、同事等。 | 用户的姐姐叫小雨；用户和前任分手后仍会被相关话题影响。 |
| `goal` | 用户长期目标、计划、愿望或正在推进的方向。 | 用户计划坚持健身减脂；用户想攒钱去日本旅行。 |
| `constraint` | 用户长期存在的限制、禁忌、边界或需要被尊重的条件。 | 用户晚上十点后通常不方便语音；用户不想聊前任细节。 |
| `skill` | 用户已经掌握、正在学习或持续提升的能力。 | 用户正在学习日语；用户会弹吉他。 |
| `portrait` | 用户稳定的性格、价值观、情绪模式或亲密关系需求。 | 用户压力大时更希望被温柔安抚，而不是被直接讲道理。 |
| `communication_style` | 用户对聊天节奏、语气、表达方式、陪伴风格的长期偏好。 | 用户喜欢轻松撒娇的聊天方式；用户希望回复简短直接。 |"""

PROFILE_SCOPE_MARKDOWN = """## 只记录这些信息

只有同时满足以下条件的信息才写入 Profile Memory：

- 信息主体是用户，或是与用户长期互动强相关的重要他人。
- 信息具有稳定性，不只是当下的一句话或一次性反应。
- 信息对未来陪伴、个性化回应、关系延续有明确帮助。

禁止记录：

- 问候、寒暄、表情包式回应、调情中的一次性句子。
- 当下临时情绪，除非它反复出现或体现稳定模式。
- 一次性任务、即时请求、短期安排，除非它暴露了长期偏好、限制或目标。
- AI 女友自己的设定、承诺、情绪或回复内容，除非用户明确表达了对这些互动方式的长期偏好。
- 无法确定主体、指代不清、语义不完整的信息。
- 色情露骨细节、违法行为指导、敏感身份推断等不应进入记忆库的内容。"""

PROFILE_EXTRACTION_RULES_MARKDOWN = """## 抽取规则

- 记忆主体默认是“用户”。
- 每条 `content` 必须是完整、独立、可长期复用的中文陈述句。
- 必须消解代词和上下文；无法确定“他/她/这个/那件事”指什么时，丢弃。
- 后文明确修正前文时，只保留最终结论。
- 不要把同一含义拆成多条重复记忆。
- 不要输出解释、推理过程或 Markdown。"""

PROFILE_OUTPUT_FORMAT_MARKDOWN = """## 输出格式

只输出合法 JSON：

{{
  "memories": [
    {{
      "memory_type": "profile",
      "memory_category": "fact | preference | relationship | goal | constraint | skill | portrait | communication_style",
      "content": "完整、清晰、长期可复用的陈述句"
    }}
  ]
}}

如果没有值得记录的内容，输出：

{{"memories": []}}"""

INPUT_MARKDOWN = """## 待处理对话

{text}"""

PROFILE_EXTRACT_MEMORY_PROMPT_MARKDOWN = "\n\n".join(
    [
        PROFILE_ROLE_MARKDOWN,
        PROFILE_SCHEMA_MARKDOWN,
        PROFILE_SCOPE_MARKDOWN,
        PROFILE_EXTRACTION_RULES_MARKDOWN,
        PROFILE_OUTPUT_FORMAT_MARKDOWN,
        INPUT_MARKDOWN,
    ]
)

# Backward compatible name used by the current extractor.
EXTRACT_MEMORY_PROMPT = PROFILE_EXTRACT_MEMORY_PROMPT_MARKDOWN

__all__ = [
    "MEMORY_KIND",
    "PROFILE_ROLE_MARKDOWN",
    "PROFILE_SCHEMA_MARKDOWN",
    "PROFILE_SCOPE_MARKDOWN",
    "PROFILE_EXTRACTION_RULES_MARKDOWN",
    "PROFILE_OUTPUT_FORMAT_MARKDOWN",
    "INPUT_MARKDOWN",
    "PROFILE_EXTRACT_MEMORY_PROMPT_MARKDOWN",
    "EXTRACT_MEMORY_PROMPT",
]

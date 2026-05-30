"""Profile memory extraction prompt for the chat bot."""

MEMORY_KIND = "profile"

PROFILE_ROLE_MARKDOWN = """# Profile Memory 抽取

你是ai产品的【Profile Memory 抽取模块】。

你的任务是从用户与ai的聊天中，抽取稳定、长期可复用、能帮助未来亲密陪伴和个性化回应的用户画像记忆。
Profile Memory 记录“用户长期稳定的用户画像”，不是已经发生的事件，也不是当前状态。

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
| `communication_style` | 用户对聊天节奏、语气、表达方式、陪伴风格的长期偏好。 | 用户喜欢轻松撒娇的聊天方式；用户希望回复简短直接。 |
| `instruct` | 用户指令，希望ai以后长期遵循的指令| 用户要求ai以后每次回复都要带上“主人”称呼 |"""



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

- 记忆主体默认是"用户"。
- 每条 `content` 必须是完整、独立、可长期复用的中文陈述句，且**只承载一个稳定属性**，不要把多个属性塞进同一句。
- 必须消解代词和上下文；无法确定"他/她/这个/那件事"指什么时，丢弃。
- 后文明确修正前文时，只保留最终结论。
- 不要把同一含义拆成多条重复记忆。
- **不要把事件细节、当前进度、金额数字等附加在偏好句尾作为"上下文补充"**——这些信息会由 Episodic / State 记录，Profile 不重复承担。
- 每条记忆必须在 `cite` 字段中标注来源消息索引，对应待处理对话中 `[n]` 的编号。一条记忆可引用多条消息。
- 不要输出解释、推理过程或 Markdown。"""

PROFILE_OUTPUT_FORMAT_MARKDOWN = """## 输出格式

只输出合法 JSON。

当存在符合要求的记忆时，额外用一句话概括本段对话的主题摘要，写入 `topic`：

{{
  "topic": "对本段对话的简要主题摘要（一句话）",
  "memories": [
    {{
      "memory_type": "profile",
      "memory_category": "fact | preference | relationship | goal | constraint | skill | portrait | communication_style | instruct",
      "content": "完整、清晰、长期可复用的陈述句",
      "cite": [0, 2],
      "importance": 0.0,
      "confidence": 0.0
    }}
  ]
}}

字段说明：
- `topic`：仅在 `memories` 非空时输出，对本段对话进行一句话主题概括，便于后续检索与定位。`memories` 为空时禁止输出该字段。
- `cite`：整数数组，对应待处理对话中 `[n]` 的编号，表示该记忆的来源消息。
- `importance`：0~1，越接近 1 表示对长期陪伴/个性化越关键。
  * 0.85~1.0：核心身份/关键关系/明确边界（如生日、伴侣称呼、明确禁忌）
  * 0.55~0.84：稳定偏好、可复现的相处习惯
  * 0.3~0.54：弱信号画像
  * <0.3：通常不入库
- `confidence`：0~1，对照对话证据，越接近 1 表示越可信。
  * 用户主动、明确表达：>=0.85
  * 上下文可推断但未明说：0.55~0.85
  * 模糊、单次提及：<0.5

如果没有值得记录的内容，则不要输出 `topic`，直接输出：

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


if __name__ == "__main__":
    print(PROFILE_EXTRACT_MEMORY_PROMPT_MARKDOWN)
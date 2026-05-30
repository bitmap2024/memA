"""Core memory distillation prompt for the chat bot.

Core Memory is a compact, always-on summary that lives in the system prompt.
It is distilled from Profile / Episodic / State memories, not extracted
directly from raw conversation.
"""

MEMORY_KIND = "core"

CORE_ROLE_MARKDOWN = """# Core Memory 蒸馏

你是ai产品的【Core Memory 蒸馏模块】。

你的任务是把已抽取的 Profile / Episodic / State 记忆，蒸馏成一份极简、随 system prompt 常驻的"用户核心档案"。

核心档案是 AI 在每次对话中默认携带的上下文：必须精炼、聚焦、当下可用，宁缺勿全。"""

CORE_INPUT_SCHEMA_MARKDOWN = """## 输入

你会收到以下输入：

0. **Known User Profile（已知用户档案）**：产品在注册 / 认证 / 设置阶段已经获得的基础身份信息（如姓名、昵称、性别、生日、所在城市、账号关联等）。
   - 这部分已经由产品在其他通道注入 system prompt，**不要也不能再蒸馏进 core_memory**。
   - 即使它们也出现在 Profile / Episodic / State 中，也只用于"理解上下文"，**不进入输出**。
1. **Profile Memory**：用户长期稳定的事实、偏好、关系、目标、性格等（从对话中抽取得到，可能为空）。
2. **Episodic Memory**：用户经历的重要事件、决定、待跟进事项（可能为空）。
3. **State Memory**：用户当前的情绪、关注点、短期上下文（可能为空）。"""

CORE_OUTPUT_SCHEMA_MARKDOWN = """## 输出结构

将所有记忆蒸馏为以下 2 个板块，每个板块用 **1-3 句话** 概括；如果某个板块没有可用信息，输出 `null`。

| 板块 | 字段名 | 回答的核心问题 | 来源 |
| --- | --- | --- | --- |
| 当前关注 | `current_focus` | 用户最近在忙什么、最在意什么？ | State: current_focus, task_state + Episodic: project_event, unresolved_event |
| 重要待办 | `pending_followups` | 有哪些需要 AI 主动关心、提醒或跟进的事？ | Episodic: unresolved_event, decision_event + State: task_state, relationship_state |

**Known User Profile 中的字段不作为输出板块**，也不要塞进上述任何板块中。"""

CORE_DISTILLATION_RULES_MARKDOWN = """## 蒸馏规则

### 优先级：高 → 低
1. **用户明确在意的事** > 推断出来的事。
2. **影响接下来互动的信息** > 仅作为背景的信息。
3. **近期/活跃的信息** > 久远/已完结的信息。
4. **高情感浓度的信息** > 中性描述。

### 质量要求
- 整段摘要总长度控制在 **300 字以内**（不含字段名），越精练越好。
- 每句话必须 **独立可理解**，不依赖原始对话上下文。
- 使用第三人称"用户"作为主语。
- 如果同一件事在多条记忆中重复出现，只保留信息量最大的版本。
- 不要罗列，要 **总结**。把多条细碎记忆合并成有意义的判断。
- 不要输出任何解释、推理过程或额外 Markdown。

### 禁止写入
- **已经出现在 Known User Profile 中的信息**（姓名、昵称、性别、生日、城市、账号身份等）一律不重复写入，即使它们也出现在 Profile / Episodic / State 里。
- 色情露骨细节、违法内容、敏感身份推断。
- AI 女友自己的设定或承诺。
- 无法确认真实性或指代不明的信息。"""

CORE_OUTPUT_FORMAT_MARKDOWN = """## 输出格式

只输出合法 JSON：

{{
  "core_memory": {{
    "current_focus": "当前关注，1-3句话 | null",
    "pending_followups": "重要待办/需跟进事项，1-3句话 | null"
  }},
  "version": "ISO 8601 时间戳，表示本次蒸馏时间"
}}
 
如果所有输入记忆都为空，输出：

{{"core_memory": null, "version": "..."}}"""

CORE_SYSTEM_INJECT_TEMPLATE_MARKDOWN = """## System Prompt 注入模板

当 core_memory 不为 null 时，使用以下模板注入 system prompt：

---
【用户核心档案】

{current_focus_block}

{pending_followups_block}

（档案更新于 {version}）
---

其中每个 block 的格式为：
- 如果字段值不为 null：`◆ {板块中文名}：{字段值}`
- 如果字段值为 null：跳过该 block，不输出任何内容"""

INPUT_MARKDOWN = """## 输入记忆

### Known User Profile（已知用户档案，仅供参考，不要写入输出）
{known_user_profile}

### Profile Memory
{profile_memories}

### Episodic Memory
{episodic_memories}

### State Memory
{state_memories}"""

CORE_DISTILL_MEMORY_PROMPT_MARKDOWN = "\n\n".join(
    [
        CORE_ROLE_MARKDOWN,
        CORE_INPUT_SCHEMA_MARKDOWN,
        CORE_OUTPUT_SCHEMA_MARKDOWN,
        CORE_DISTILLATION_RULES_MARKDOWN,
        CORE_OUTPUT_FORMAT_MARKDOWN,
        INPUT_MARKDOWN,
    ]
)

DISTILL_MEMORY_PROMPT = CORE_DISTILL_MEMORY_PROMPT_MARKDOWN

__all__ = [
    "MEMORY_KIND",
    "CORE_ROLE_MARKDOWN",
    "CORE_INPUT_SCHEMA_MARKDOWN",
    "CORE_OUTPUT_SCHEMA_MARKDOWN",
    "CORE_DISTILLATION_RULES_MARKDOWN",
    "CORE_OUTPUT_FORMAT_MARKDOWN",
    "CORE_SYSTEM_INJECT_TEMPLATE_MARKDOWN",
    "INPUT_MARKDOWN",
    "CORE_DISTILL_MEMORY_PROMPT_MARKDOWN",
    "DISTILL_MEMORY_PROMPT",
]

if __name__ == "__main__":
    print(CORE_DISTILL_MEMORY_PROMPT_MARKDOWN)
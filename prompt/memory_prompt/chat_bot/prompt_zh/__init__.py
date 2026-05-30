"""Chat bot prompt pack: profile / episodic / state / core / update / merge."""

from prompt.memory_prompt.chat_bot.prompt_zh.profile import EXTRACT_MEMORY_PROMPT as PROFILE_EXTRACT_PROMPT
from prompt.memory_prompt.chat_bot.prompt_zh.episodic import EXTRACT_MEMORY_PROMPT as EPISODIC_EXTRACT_PROMPT
from prompt.memory_prompt.chat_bot.prompt_zh.state import EXTRACT_MEMORY_PROMPT as STATE_EXTRACT_PROMPT
from prompt.memory_prompt.chat_bot.prompt_zh.memory_update_merge import (
    MERGE_MEMORY_PROMPT,
    UPDATE_MEMORY_PROMPT,
    CATEGORY_DOC_PROMPT,
)

__all__ = [
    "PROFILE_EXTRACT_PROMPT",
    "EPISODIC_EXTRACT_PROMPT",
    "STATE_EXTRACT_PROMPT",
    "MERGE_MEMORY_PROMPT",
    "UPDATE_MEMORY_PROMPT",
    "CATEGORY_DOC_PROMPT",
]

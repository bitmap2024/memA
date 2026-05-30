"""pytest 共享 fixture（src/test 下所有测试自动加载）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 保证从项目根目录可 import config / src
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def real_text_compressor():
    """集成测试用：加载真实 LLMLingua-2 模型；缺依赖或模型则 skip。"""
    try:
        import llmlingua  # noqa: F401
    except ImportError:
        pytest.skip("llmlingua 未安装")

    from config.setting import Config
    from src.extract_mem.text_compressor import TextCompressor

    model_path = Config.compressor.MODEL_PATH
    if not Path(model_path).exists():
        pytest.skip(f"模型路径不存在: {model_path}")

    return TextCompressor(
        model_path=model_path,
        max_tokens=Config.compressor.MAX_TOKENS,
    )

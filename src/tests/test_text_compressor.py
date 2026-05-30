"""TextCompressor 单元测试与可选集成测试。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.extract_mem.text_compressor import TextCompressor


@pytest.fixture
def mock_prompt_compressor():
    """Mock PromptCompressor，避免加载真实模型。"""
    with patch("src.extract_mem.text_compressor.PromptCompressor") as cls:
        instance = MagicMock()
        cls.return_value = instance
        yield cls, instance


class TestTextCompressorInit:
    def test_raises_when_llmlingua_missing(self):
        with patch("src.extract_mem.text_compressor.PromptCompressor", None):
            with pytest.raises(RuntimeError, match="llmlingua"):
                TextCompressor(model_path="/fake/model")

    def test_init_stores_max_tokens_and_uses_cpu_without_torch(
        self, mock_prompt_compressor
    ):
        mock_cls, _ = mock_prompt_compressor
        with patch("src.extract_mem.text_compressor.torch", None):
            tc = TextCompressor(model_path="/fake/model", max_tokens=256)

        assert tc.max_tokens == 256
        mock_cls.assert_called_once_with(
            model_name="/fake/model",
            use_llmlingua2=True,
            device_map="cpu",
        )

    def test_init_uses_cuda_when_available(self, mock_prompt_compressor):
        mock_cls, _ = mock_prompt_compressor
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch("src.extract_mem.text_compressor.torch", mock_torch):
            TextCompressor(model_path="/fake/model")

        mock_cls.assert_called_once_with(
            model_name="/fake/model",
            use_llmlingua2=True,
            device_map="cuda:0",
        )

    def test_init_falls_back_to_cpu_on_cuda_check_error(
        self, mock_prompt_compressor
    ):
        mock_cls, _ = mock_prompt_compressor
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.side_effect = RuntimeError("cuda error")

        with patch("src.extract_mem.text_compressor.torch", mock_torch):
            TextCompressor(model_path="/fake/model")

        mock_cls.assert_called_once_with(
            model_name="/fake/model",
            use_llmlingua2=True,
            device_map="cpu",
        )


class TestCompress:
    def test_compress_delegates_with_defaults(self, mock_prompt_compressor):
        _, instance = mock_prompt_compressor
        expected = {"compressed_prompt": "short"}
        instance.compress_prompt_llmlingua2.return_value = expected

        tc = TextCompressor(model_path="/fake/model")
        result = tc.compress("hello world", rate=0.3)

        instance.compress_prompt_llmlingua2.assert_called_once_with(
            "hello world",
            rate=0.3,
            force_tokens=TextCompressor.DEFAULT_FORCE_TOKENS,
            chunk_end_tokens=TextCompressor.DEFAULT_CHUNK_END_TOKENS,
            return_word_label=True,
            drop_consecutive=True,
        )
        assert result == expected

    def test_compress_passes_custom_options(self, mock_prompt_compressor):
        _, instance = mock_prompt_compressor
        instance.compress_prompt_llmlingua2.return_value = {}
        force_tokens = ["a"]
        chunk_end_tokens = ["b"]

        tc = TextCompressor(model_path="/fake/model")
        tc.compress(
            "text",
            rate=0.2,
            force_tokens=force_tokens,
            chunk_end_tokens=chunk_end_tokens,
            return_word_label=False,
            drop_consecutive=False,
        )

        instance.compress_prompt_llmlingua2.assert_called_once_with(
            "text",
            rate=0.2,
            force_tokens=force_tokens,
            chunk_end_tokens=chunk_end_tokens,
            return_word_label=False,
            drop_consecutive=False,
        )


class TestGetAnnotatedResults:
    @staticmethod
    def _compressor_without_init() -> TextCompressor:
        return TextCompressor.__new__(TextCompressor)

    def test_maps_label_one_to_plus_and_zero_to_minus(self):
        tc = self._compressor_without_init()
        results = {
            "fn_labeled_original_prompt": "hello 1\t\t|\t\tworld 0",
        }

        assert tc.get_annotated_results(results) == [
            ("hello", "+"),
            ("world", "-"),
        ]

    def test_supports_custom_separators(self):
        tc = self._compressor_without_init()
        results = {"fn_labeled_original_prompt": "foo:1||bar:0"}

        assert tc.get_annotated_results(results, word_sep="||", label_sep=":") == [
            ("foo", "+"),
            ("bar", "-"),
        ]


class TestCompressAndAnnotate:
    def test_empty_or_whitespace_text_skips_compression(
        self, mock_prompt_compressor
    ):
        _, instance = mock_prompt_compressor
        tc = TextCompressor(model_path="/fake/model")

        assert tc.compress_and_annotate("") == (None, [])
        assert tc.compress_and_annotate("   \n\t  ") == (None, [])
        instance.compress_prompt_llmlingua2.assert_not_called()

    def test_returns_results_and_annotations(self, mock_prompt_compressor):
        _, instance = mock_prompt_compressor
        results = {"fn_labeled_original_prompt": "hi 1\t\t|\t\tthere 0"}
        instance.compress_prompt_llmlingua2.return_value = results

        tc = TextCompressor(model_path="/fake/model")
        out_results, annotated = tc.compress_and_annotate("hello there", rate=0.4)

        assert out_results == results
        assert annotated == [("hi", "+"), ("there", "-")]
        instance.compress_prompt_llmlingua2.assert_called_once()


@pytest.mark.integration
def test_real_compress_and_annotate(real_text_compressor):
    """需要本地模型与 llmlingua；缺依赖时 conftest 自动 skip。"""
    text = "用户喜欢喝拿铁，助手建议减少糖分摄入。"
    results, annotated = real_text_compressor.compress_and_annotate(text, rate=0.5)

    assert results is not None
    assert isinstance(results, dict)
    assert "fn_labeled_original_prompt" in results
    assert isinstance(annotated, list)
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in annotated)
    assert all(label in ("+", "-") for _, label in annotated)


# # 仅单元测试（不依赖模型）
# python -m pytest src/test/test_text_compressor.py -m "not integration" -v

# # 含集成测试（需 llmlingua + 本地模型路径）
# python -m pytest src/test/test_text_compressor.py -v
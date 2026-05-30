from typing import List, Tuple, Dict, Any, Optional


def _patch_tiktoken_offline() -> None:
    """拦截 llmlingua 初始化时对 tiktoken 的联网调用。

    llmlingua 的 PromptCompressor.__init__ 无条件执行
    `tiktoken.encoding_for_model("gpt-3.5-turbo")`，会去联网下载
    OpenAI 的 cl100k_base.tiktoken。国内访问
    openaipublic.blob.core.windows.net 会被重置/超时而报错。

    在 llmlingua2 流程里该 tokenizer 仅用于统计 token 数量
    （compressed_tokens / ratio / saving 等指标），不影响压缩结果，
    与本地 BERT 模型自带的分词器无关。这里把它替换成一个不联网的
    近似计数器，彻底避免下载。
    """
    try:
        import tiktoken  # type: ignore
    except Exception:
        return

    class _ApproxEncoding:
        """不联网的近似编码器，仅提供 token 计数能力。"""

        name = "approx_offline"

        @staticmethod
        def _tokenize(text: str) -> List[str]:
            tokens: List[str] = []
            buf = ""
            for ch in text:
                if ch.isspace():
                    if buf:
                        tokens.append(buf)
                        buf = ""
                elif ord(ch) > 0x2E80 or not ch.isalnum():
                    if buf:
                        tokens.append(buf)
                        buf = ""
                    tokens.append(ch)
                else:
                    buf += ch
            if buf:
                tokens.append(buf)
            return tokens

        def encode(self, text: str, *args, **kwargs) -> List[int]:
            return list(range(len(self._tokenize(text))))

        def decode(self, tokens, *args, **kwargs) -> str:
            return ""

    def _offline_encoding_for_model(model_name: str, *args, **kwargs):
        return _ApproxEncoding()

    tiktoken.encoding_for_model = _offline_encoding_for_model  # type: ignore


_patch_tiktoken_offline()

try:
    from llmlingua import PromptCompressor  # type: ignore
except Exception:  # pragma: no cover - optional heavy dependency
    PromptCompressor = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


class TextCompressor:
    """基于 LLMLingua-2 的文本压缩器"""
    
    DEFAULT_FORCE_TOKENS = ['\n', '.', '!', '?', ',', '、']
    DEFAULT_CHUNK_END_TOKENS = ['.', '\n']
    
    def __init__(self, model_path, max_tokens: int = 512):
        """
        初始化压缩器
        
        Args:
            model_path: 模型路径，默认使用 DEFAULT_MODEL_PATH
            max_tokens: 压缩后的最大 token 数，默认 512
        """
        self.max_tokens = max_tokens
        if PromptCompressor is None:
            raise RuntimeError(
                "llmlingua 未安装，请先 `pip install llmlingua` 或在 .env 中关闭压缩模型路径"
            )
        device_map = "cpu"
        if torch is not None:
            try:
                device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
            except Exception:
                device_map = "cpu"
        self.compressor = PromptCompressor(
            model_name=model_path,
            use_llmlingua2=True,
            device_map=device_map,
        )
    
    def compress(
        self,
        text: str,
        rate: float = 0.8,
        force_tokens: Optional[List[str]] = None,
        chunk_end_tokens: Optional[List[str]] = None,
        return_word_label: bool = True,
        drop_consecutive: bool = True
    ) -> Dict[str, Any]:
        """
        压缩文本
        
        Args:
            text: 原始文本
            rate: 压缩率 (0-1)，越小压缩越多
            force_tokens: 强制保留的 token 列表
            chunk_end_tokens: 分块结束 token 列表
            return_word_label: 是否返回词标签
            drop_consecutive: 是否丢弃连续重复
            
        Returns:
            包含压缩结果的字典
        """
        force_tokens = force_tokens or self.DEFAULT_FORCE_TOKENS
        chunk_end_tokens = chunk_end_tokens or self.DEFAULT_CHUNK_END_TOKENS
        
        results = self.compressor.compress_prompt_llmlingua2(
            text,
            rate=rate,
            force_tokens=force_tokens,
            chunk_end_tokens=chunk_end_tokens,
            return_word_label=return_word_label,
            drop_consecutive=drop_consecutive
        )
        return results
    
    def compress_and_annotate(
        self,
        text: str,
        rate: float = 0.8,
        **kwargs
    ) -> Tuple[Dict[str, Any], List[Tuple[str, str]]]:
        """
        压缩文本并返回标注结果
        
        Args:
            text: 原始文本
            rate: 压缩率
            **kwargs: 传递给 compress() 的其他参数
            
        Returns:
            (压缩结果字典, 标注结果列表)
        """
        if not text or not text.strip():
            return None, []
        results = self.compress(text, rate=rate, **kwargs)
        return results

if __name__ == "__main__":
    compressor = TextCompressor(model_path="d:/aiworks/premodel/llmlingua-2-bert-base-multilingual-cased-meetingbank")
    text = "咱们先把这些航班订好吧！接下来，您能否推荐一些坦帕的住宿，位置便利，方便游览景点，并且符合我单人入住的预算？"
    results = compressor.compress(text, rate=0.5)
    print("results:", results)
    print("-"*100)
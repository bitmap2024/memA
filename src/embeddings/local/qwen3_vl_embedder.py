"""Qwen3-VL embedding 推理封装。

提供 :class:`Qwen3VLEmbedder` 类，基于 ``sentence-transformers`` 加载
``Qwen/Qwen3-VL-Embedding-2B``，支持纯文本、纯图像 (URL / 本地路径) 以及
text+image 混合三种输入形态的统一编码。

该类不关心 gRPC / 网络层，只负责模型加载和前向计算，方便被 server 复用。
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer


DenseVec = np.ndarray

# 单条输入支持的形态:
#   - str: 纯文本 或 图像 url / 本地路径 (Qwen3-VL 内部按 prefix 区分)
#   - dict: {"text": str, "image": str} 混合输入
EmbInput = Union[str, Dict[str, str]]


class Qwen3VLEmbedder:
    """对 ``Qwen/Qwen3-VL-Embedding-2B`` 的轻量封装。

    设计目标:
        - 一次性加载模型, 提供线程安全的 ``encode`` 接口
        - 输入既可以是单条 (``str`` / ``dict``), 也可以是 ``List``;
          输出统一是 ``List[np.ndarray]`` (长度 = 输入长度),
          方便 server 端 dynamic batch 后按位拆分回各请求
        - 同时暴露 ``similarity`` 方法, 与底层 SentenceTransformer 行为一致
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-VL-Embedding-2B",
        device: Optional[str] = "cuda:0",
        batch_size: int = 8,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = True,
        warmup: bool = True,
        **kwargs: Any,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.default_batch_size = batch_size

        logger.info(
            "loading Qwen3-VL from {} on {} (batch_size={})",
            model_path, device, batch_size,
        )
        self.model = SentenceTransformer(
            model_path,
            device=device,
            cache_folder=cache_dir if cache_dir is not None else os.getenv("HF_HUB_CACHE"),
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        # SentenceTransformer 内部使用 transformers 模型, 前向时会修改 attribute,
        # 用大锁串行化, 防止多线程下竞争.
        self._lock = threading.Lock()

        if warmup:
            try:
                self.encode(["warmup"])
                logger.info("Qwen3-VL warmup done")
            except Exception:  # pragma: no cover
                logger.exception("warmup failed (will continue anyway)")

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def encode(
        self,
        inputs: Union[EmbInput, Sequence[EmbInput]],
        batch_size: Optional[int] = None,
        normalize_embeddings: bool = False,
        show_progress_bar: bool = False,
        **kwargs: Any,
    ) -> List[DenseVec]:
        """批量编码文本 / 图像 / 文本+图像。

        Args:
            inputs: 单条输入或 List。每条输入可以是:

                - ``str``: 文本 或 图像 URL / 本地路径
                - ``dict``: ``{"text": str, "image": str}`` 混合形态
            batch_size: 模型内部 mini-batch 大小, 不传则用默认
            normalize_embeddings: 是否对输出做 L2 归一化
            show_progress_bar: 是否打印进度条
            **kwargs: 透传给 ``SentenceTransformer.encode``

        Returns:
            长度与输入相同的 ``List[np.ndarray]``。每个向量 shape 为 ``(D,)``,
            Qwen3-VL-Embedding-2B 下 ``D == 2048``。
        """
        single_input = isinstance(inputs, (str, dict))
        item_list: List[EmbInput] = [inputs] if single_input else list(inputs)  # type: ignore[list-item]
        if len(item_list) == 0:
            return []

        bs = batch_size or self.default_batch_size

        with self._lock:
            arr = self.model.encode(
                item_list,
                batch_size=bs,
                normalize_embeddings=normalize_embeddings,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
                **kwargs,
            )

        return [arr[i] for i in range(arr.shape[0])]

    def similarity(
        self,
        query_embeddings: Union[Sequence[DenseVec], np.ndarray],
        doc_embeddings: Union[Sequence[DenseVec], np.ndarray],
    ) -> Any:
        """计算 query 与 doc 之间的相似度矩阵。

        直接转发给底层 ``SentenceTransformer.similarity``,
        返回值与 SentenceTransformer 一致 (通常为 ``torch.Tensor``)。
        """
        q = np.stack(query_embeddings) if not isinstance(query_embeddings, np.ndarray) else query_embeddings
        d = np.stack(doc_embeddings) if not isinstance(doc_embeddings, np.ndarray) else doc_embeddings
        return self.model.similarity(q, d)


# ---------------------------------------------------------------------- #
# 旧脚本风格的自检入口, 直接 `python qwen3_vl_embedder.py` 跑一遍
# ---------------------------------------------------------------------- #
def _self_test() -> None:
    embedder = Qwen3VLEmbedder()

    queries = [
        "A woman playing with her dog on a beach at sunset.",
        "Pet owner training dog outdoors near water.",
        "Woman surfing on waves during a sunny day.",
        "City skyline view from a high-rise building at night.",
    ]

    documents: List[EmbInput] = [
        "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, "
        "as the dog offers its paw in a heartwarming display of companionship and trust.",
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        {
            "text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, "
                    "as the dog offers its paw in a heartwarming display of companionship and trust.",
            "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        },
    ]

    query_embeddings = embedder.encode(queries)
    doc_embeddings = embedder.encode(documents)
    print(np.stack(query_embeddings).shape, np.stack(doc_embeddings).shape)

    similarities = embedder.similarity(query_embeddings, doc_embeddings)
    print(similarities)


if __name__ == "__main__":
    _self_test()

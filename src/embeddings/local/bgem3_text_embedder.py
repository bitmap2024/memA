"""BGE-M3 embedding 推理封装。

提供 `BGEM3Embedder` 类，同时返回 dense / sparse 向量，并支持 batch 编码。
该类不关心 gRPC / 网络层，只负责模型加载和前向计算，方便被 server 复用。
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from FlagEmbedding import BGEM3FlagModel


from loguru import logger


SparseDict = Dict[int, float]
DenseVec = np.ndarray


class BGEM3TextEmbedder:
    """对 :class:`BGEM3FlagModel` 的轻量封装。

    设计目标:
        - 一次性加载模型, 提供线程安全的 ``encode`` 接口
        - 同时支持 dense / sparse, 调用方按需取
        - 输入既可以是单条 ``str``, 也可以是 ``List[str]``;
          输出统一是 ``list``(长度 = 输入长度), 方便 server 端 dynamic batch
          后按位拆分回各请求
        - lexical_weights 的 key 从 ``str(token_id)`` 归一化为 ``int``,
          直接对应 proto 中 ``map<uint32, float>``
    """

    def __init__(
        self,
        model_path: str = "D:/aiworks/premodel/bge-m3",
        device: str = "cuda:0",
        pooling_method: str = "cls",
        use_fp16: bool = True,
        max_length: int = 8192,
        batch_size: int = 32,
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.default_max_length = max_length
        self.default_batch_size = batch_size

        logger.info(
            "loading BGE-M3 from %s on %s (fp16=%s, pooling=%s)",
            model_path, device, use_fp16, pooling_method,
        )
        self.model = BGEM3FlagModel(
            model_path,
            devices=device,
            pooling_method=pooling_method,
            use_fp16=use_fp16,
            cache_dir=cache_dir if cache_dir is not None else os.getenv("HF_HUB_CACHE"),
            **kwargs,
        )

        # BGEM3FlagModel 内部状态非线程安全(前向时会写 attribute), 用大锁串行化.
        # dynamic batcher 是单 worker 线程顺序调用, 这里加锁是 belt-and-suspenders.
        self._lock = threading.Lock()

        # warmup: 先跑一次拿到 dense / sparse, 让 cuda kernel 预热
        try:
            self.encode(["warmup"], return_dense=True, return_sparse=True)
            logger.info("BGE-M3 warmup done")
        except Exception:  # pragma: no cover
            logger.exception("warmup failed (will continue anyway)")

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def encode(
        self,
        texts: Union[str, Sequence[str]],
        return_dense: bool = True,
        return_sparse: bool = True,
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """批量编码文本。

        Args:
            texts: 单条 str 或 List[str]
            return_dense: 是否返回 dense 向量
            return_sparse: 是否返回 sparse (lexical) 向量
            batch_size: 模型内部 mini-batch 大小, 不传则用默认
            max_length: 最大 token 长度, 不传则用默认

        Returns:
            ``{"dense": List[np.ndarray] | None, "sparse": List[Dict[int, float]] | None}``
            两个 list 长度与输入文本数相等; 未请求的字段为 ``None``.
        """
        if not return_dense and not return_sparse:
            raise ValueError("return_dense 与 return_sparse 不能同时为 False")

        single_input = isinstance(texts, str)
        text_list: List[str] = [texts] if single_input else list(texts)
        if len(text_list) == 0:
            return {"dense": [] if return_dense else None,
                    "sparse": [] if return_sparse else None}

        bs = batch_size or self.default_batch_size
        ml = max_length or self.default_max_length

        with self._lock:
            raw = self.model.encode(
                text_list,
                batch_size=bs,
                max_length=ml,
                return_dense=return_dense,
                return_sparse=return_sparse,
                return_colbert_vecs=False,
            )

        dense_out: Optional[List[DenseVec]] = None
        sparse_out: Optional[List[SparseDict]] = None

        if return_dense:
            dense_arr = raw["dense_vecs"]  # (N, D) numpy array
            dense_out = [dense_arr[i] for i in range(dense_arr.shape[0])]

        if return_sparse:
            sparse_out = [self._normalize_sparse(w) for w in raw["lexical_weights"]]

        return {"dense": dense_out, "sparse": sparse_out}

    # 方便外部需要时调用打分函数(与 FlagEmbedding 一致)
    def compute_lexical_matching_score(
        self,
        q_sparse: List[SparseDict],
        p_sparse: List[SparseDict],
    ) -> np.ndarray:
        # FlagEmbedding 的 compute_lexical_matching_score 接受 str-key dict,
        # 这里再转回去, 保持外部接口干净.
        q = [{str(k): v for k, v in d.items()} for d in q_sparse]
        p = [{str(k): v for k, v in d.items()} for d in p_sparse]
        return self.model.compute_lexical_matching_score(q, p)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_sparse(weights: Dict[Any, float]) -> SparseDict:
        """把 ``{str(token_id): float}`` 归一化为 ``{int: float}``。

        BGEM3 输出是 ``defaultdict(int, str -> float)``, 这里直接转 ``int`` key,
        与 proto 的 ``map<uint32, float>`` 对应.
        """
        return {int(k): float(v) for k, v in weights.items()}


# ---------------------------------------------------------------------- #
# 旧脚本风格的自检入口, 直接 `python bgem3_inferrence.py` 跑一遍
# ---------------------------------------------------------------------- #
def _self_test() -> None:
    embedder = BGEM3TextEmbedder()

    queries = ["What is BGE M3?", "Defination of BM25"]
    passages = [
        "BGE M3 is an embedding model supporting dense retrieval, "
        "lexical matching and multi-vector interaction.",
        "BM25 is a bag-of-words retrieval function that ranks a set of "
        "documents based on the query terms appearing in each document",
    ]

    q = embedder.encode(queries, return_dense=True, return_sparse=True)
    p = embedder.encode(passages, return_dense=True, return_sparse=True)
    print("dense q:", q["dense"])
    print("dense p:", p["dense"])
    print("sparse q:", q["sparse"])
    print("sparse p:", p["sparse"])
    
    dense_scores = np.stack(q["dense"]) @ np.stack(p["dense"]).T
    sparse_scores = embedder.compute_lexical_matching_score(q["sparse"], p["sparse"])

    print("Dense score:\n", dense_scores)
    print("Sparse score:\n", sparse_scores)


if __name__ == "__main__":
    _self_test()

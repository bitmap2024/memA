#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Topic segmentation for LLM chat messages.

The segmenter keeps short conversations intact. When the conversation is over
the token budget, it folds each user/assistant turn into one record, embeds the
turn records, builds a cosine similarity matrix, and splits contiguous topics
where semantic similarity drops below the configured threshold.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.path.append("D:/aiworks/code/memA")
# 中文字符按字计, 英文/数字连续段按 token 计
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_ASCII_WORD_RE = re.compile(r"[A-Za-z0-9]+")

from src.embeddings.local.bgem3_text_embedder import BGEM3TextEmbedder
EmbeddingFn = Callable[[List[str]], Any]


@dataclass
class TopicSegment:
    """One contiguous topic segment."""

    topic_id: int
    messages: List[Dict[str, Any]]
    records: List[Dict[str, Any]]
    start_record_index: int
    end_record_index: int
    token_count: int


@dataclass
class TopicSegmentResult:
    """Detailed result for topic segmentation."""

    segments: List[TopicSegment]
    similarity_matrix: List[List[float]]
    total_tokens: int
    segmented: bool


class TopicSegmenter:
    """
    Split LLM messages into topic-level contiguous chunks.

    Args:
        embedding_client: Object that exposes ``get_embeddings(List[str])``.
        embedding_fn: Optional callable alternative to ``embedding_client``.
        token_threshold: Conversations no longer than this are not segmented.
        similarity_threshold: Start a new topic when adjacent turn similarity is
            below this value.
        tokenizer: Optional tokenizer with ``encode`` method.
    """

    USER_KEYS = ("user_content", "request_content", "RequestContent", "user")
    ASSISTANT_KEYS = ("assistant_content","response_content","ResponseContent","assistant")

    def __init__(
        self,
        embedder: Optional[Any] = None,
        embedding_fn: Optional[EmbeddingFn] = None,
        token_threshold: int = 512,
        similarity_threshold: float = 0.55,
    ):
        if embedder is None and embedding_fn is None:
            logger.error("未配置 embedding_client/embedding_fn，长文本主题划分时将无法计算向量")
            raise ValueError("未配置 embedding_client/embedding_fn，长文本主题划分时将无法计算向量")
        
        self.embedder = embedder
        self.embedding_fn = embedding_fn
        self.token_threshold = max(1, int(token_threshold))
        self.similarity_threshold = float(similarity_threshold)

    def segment(self, messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Return only message chunks, suitable for callers that do not need diagnostics."""

        return [segment.messages for segment in self.segment_with_details(messages).segments]

    def segment_with_details(self, messages: List[Dict[str, Any]]) -> TopicSegmentResult:
        """
        Segment messages and return the similarity matrix plus metadata.

        ``messages`` can be either OpenAI-style role/content messages or records
        containing user/assistant content fields.
        """

        if not messages:
            return TopicSegmentResult([], [], 0, False)

        records = self.messages_to_turn_records(messages)
        # logger.debug(f"records count: {len(records)}")
        total_tokens = self.count_messages_tokens(messages)
        logger.debug(f"total_tokens: {total_tokens}")
        if total_tokens <= self.token_threshold or len(records) <= 1:
            return TopicSegmentResult(
                segments=[
                    TopicSegment(
                        topic_id=0,
                        messages=list(messages),
                        records=records,
                        start_record_index=0,
                        end_record_index=max(0, len(records) - 1),
                        token_count=total_tokens,
                    )
                ],
                similarity_matrix=[],
                total_tokens=total_tokens,
                segmented=False,
            )

        texts = [record["text"] for record in records]
        embeddings = self.embed_records(texts)
        similarity_matrix = self.compute_similarity_matrix(embeddings)
        boundaries = self.find_topic_boundaries(similarity_matrix)
        logger.debug(f"boundaries: {boundaries}")
        segments = self.build_segments(records, boundaries)
        logger.debug(f"segments count: {len(segments)}")
       
        return TopicSegmentResult(
            segments=segments,
            similarity_matrix=similarity_matrix,
            total_tokens=total_tokens,
            segmented=len(segments) > 1,
        )

    def messages_to_turn_records(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fold messages into user+assistant turn records.

        Records that already contain both sides are preserved as one turn. For
        role/content messages, a user message and following assistant message are
        combined into a single record.
        """
        records: List[Dict[str, Any]] = []
        if not messages:
            return records

        for index in range(0, len(messages) - 1, 2):
            pair = [messages[index], messages[index + 1]]
            text = (messages[index].get("content") or "") + (messages[index + 1].get("content") or "")
            records.append({"text": text, "messages": pair})

        # 兜底: messages 个数为奇数时, 单独保留最后一条, 避免静默丢失
        if len(messages) % 2 == 1:
            tail = messages[-1]
            records.append({"text": tail.get("content") or "", "messages": [tail]})
        return records

    def count_messages_tokens(self, messages: List[Dict[str, Any]]) -> int:
        return sum(self.count_text_tokens(message) for message in messages)

    def count_text_tokens(self, message: Dict[str, Any]) -> int:
        """统计一条 OpenAI 协议 message 的 token 数：中文按字，英文/数字按词。"""
        if not message:
            return 0
        return self._count_tokens_in_content(message.get("content"))

    def _count_tokens_in_content(self, content: Any) -> int:
        if content is None:
            return 0
        # OpenAI 多模态格式：content 可能是 [{"type": "text", "text": "..."}, ...]
        if isinstance(content, list):
            return sum(
                self._count_tokens_in_content(part.get("text") or part.get("content"))
                if isinstance(part, dict)
                else self._count_tokens_in_text(part)
                for part in content
            )
        if isinstance(content, str):
            return self._count_tokens_in_text(content)
        return 0

    @staticmethod
    def _count_tokens_in_text(text: str) -> int:
        if not text:
            return 0
        cjk_count = len(_CJK_CHAR_RE.findall(text))
        ascii_count = len(_ASCII_WORD_RE.findall(text))
        return cjk_count + ascii_count


    def embed_records(self, texts: List[str]) -> List[List[float]]:
        if self.embedding_fn is not None:
            embeddings = self.embedding_fn(texts)
        elif self.embedder is not None:
            embeddings = self.embedder.encode(texts)
            embeddings = embeddings["dense"]
        else:
            raise ValueError("长文本主题划分需要配置 embedding_client 或 embedding_fn")

        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()

        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise ValueError(
                f"embedding 结果维度不符合预期: expected ({len(texts)}, dim)"
            )

        normalized_embeddings: List[List[float]] = []
        for embedding in embeddings:
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            if not isinstance(embedding, list):
                raise ValueError("embedding 结果必须是二维向量")
            normalized_embeddings.append([float(value) for value in embedding])

        return normalized_embeddings

    def compute_similarity_matrix(self, embeddings: List[List[float]]) -> List[List[float]]:
        matrix = cosine_similarity(np.asarray(embeddings, dtype=np.float32))
        return matrix.tolist()

    def find_topic_boundaries(self, similarity_matrix: List[List[float]]) -> List[int]:
        """
        Return start indexes for each topic segment.

        Topic segmentation must preserve conversation order, so boundaries are
        detected from adjacent turn similarities in the full similarity matrix.
        """

        boundaries = [0]
        for index in range(len(similarity_matrix) - 1):
            adjacent_similarity = float(similarity_matrix[index][index + 1])
            if adjacent_similarity < self.similarity_threshold:
                boundaries.append(index + 1)
        return boundaries

    def build_segments(self, records: List[Dict[str, Any]], boundaries: List[int]) -> List[TopicSegment]:
        segments: List[TopicSegment] = []
        starts = boundaries or [0]

        for topic_id, start in enumerate(starts):
            end = starts[topic_id + 1] if topic_id + 1 < len(starts) else len(records)
            segment_records = records[start:end]
            segment_messages = [
                message
                for record in segment_records
                for message in record.get("messages", [])
            ]
            token_count = sum(self.count_text_tokens(message) for message in segment_messages)
            segments.append(
                TopicSegment(
                    topic_id=topic_id,
                    messages=segment_messages,
                    records=segment_records,
                    start_record_index=start,
                    end_record_index=end - 1,
                    token_count=token_count,
                )
            )

        return segments


    # ------------------------------------------------------------------
    # 调试辅助
    # ------------------------------------------------------------------
    @staticmethod
    def format_segment(segment: "TopicSegment") -> str:
        """将一个 TopicSegment 格式化为可读字符串，格式示例：
        [0] speaker:user,content:...
        [1] speaker:ai,content:...
        """
        _ROLE_ALIAS = {"assistant": "ai", "system": "system"}
        lines = []
        for idx, msg in enumerate(segment.messages):
            role = msg.get("role", "user").lower()
            speaker = _ROLE_ALIAS.get(role, role)
            content = (msg.get("content") or "").replace("\n", "\\n")
            lines.append(f"[{idx}] speaker:{speaker},content:{content}")
        return "\n".join(lines)

    def print_segments(self, result: "TopicSegmentResult") -> None:
        """打印 segment_with_details 的结果，按主题分块输出。"""
        print(f"总 token 数: {result.total_tokens}  |  是否切分: {result.segmented}  |  主题数: {len(result.segments)}")
        for seg in result.segments:
            print(f"\n{'='*60}")
            print(f"主题 {seg.topic_id}  records:{seg.start_record_index}-{seg.end_record_index}  tokens:{seg.token_count}")
            print(f"{'='*60}")
            print(self.format_segment(seg))

if __name__ == "__main__":
    messages = [
    {"role": "user", "content": "我打算在今年十一月带父母去京都赏枫，一共五天四夜。父母走路较慢，希望行程宽松，每天不超过两个主景点，必去伏见稻荷、清水寺、岚山竹林和锦市场，还想体验一晚传统町家民宿留纪念。"}     ,
    {"role": "assistant", "content": "建议前两天住京都站附近，方便去伏见稻荷和东福寺；后两天移到岚山或祇园，游竹林小径、渡月桥和清水寺。锦市场可安排在午餐时段顺路逛，坡道多处以休息为主，必要时改乘出租车方便步行。"},
    {"role": "user", "content": "母亲有关节旧伤，预订町家时要确认是否有独立卫生间、楼梯层数，以及能否提供席式卧房。十一月京都日均十五度左右，建议层叠穿衣并备轻便羽绒与围巾，父亲需随身带常用药品备用。"},
    {"role": "assistant", "content": "从关西机场可乘Haruka特急直达京都站，对长辈最省力。父亲喜欢摄影，推荐南禅寺三门与圆山公园清晨时段；永观堂夜枫和瑠璃光院需提前预约并关注放票时间，也可半日去宇治喝茶。"},
    {"role": "user", "content": "家里有人偏素食，也想安排半天茶道或和菓子制作体验，节奏不要太赶，体验课最好避开周末高峰。锦市场部分摊位只收现金，是否应在京都站提前兑换适量日元备用？若下雨是否有室内备选方案？"},
    {"role": "assistant", "content": "祇园周边有多家茶室与和菓子教室，半日体验通常含讲解与品尝，建议提前一周预约。素食可选汤豆腐与精进料理，预订时说明饮食限制，随身携带少量现金，返回时可买御守与和纸纪念品。"},
    {"role": "user", "content": "最近我想系统学习Python，方向是数据分析与办公自动化。目前只会Excel函数和透视表，没有编程基础，希望两三个月内能自动清洗周报、合并多表、出图并发邮件。"},
    {"role": "assistant", "content": "建议先掌握变量、条件、循环、函数及列表字典，再学习pandas、openpyxl和matplotlib。每天一到两小时，约十到十二周可完成常规报表流水线，遇错先读traceback。"},
    {"role": "user", "content": "办公Windows用Miniconda还是python.org加venv更合适？pandas里merge和concat如何区分？如何把图表嵌入Excel？定时任务能否后台运行脚本？"},
    {"role": "assistant", "content": "推荐Miniconda或Python加venv管理依赖。merge按键关联两表，concat拼接同结构表。openpyxl可写入matplotlib图片，计划程序可定时调用脚本。"},
    {"role": "user", "content": "想做一个练手项目：每月自动汇总三地销售Excel，按产品类别生成柱状图并写入PPT。遇到空值、重复订单号和日期格式不统一时，应如何设计清洗流程与日志记录？"},
    {"role": "assistant", "content": "可按读取、校验、清洗、聚合、出图、写PPT拆成函数，主流程用main串联。统一日期格式，按订单号去重，空值按规则处理并写日志，requirements锁定版本便于同事维护。"},
    ]
    hanzi_count = sum(len(message["content"]) for message in messages)
    print(f"对话汉字总数: {hanzi_count}")
    
    embedder = BGEM3TextEmbedder(
        model_path="D:/aiworks/premodel/bge-m3",
        device="cuda:0",
        pooling_method="cls",
        use_fp16=True,
        max_length=8192,
        batch_size=32,
        cache_dir=None,
    )
    
    segmenter = TopicSegmenter(
        embedder=embedder,
        token_threshold=512,
        similarity_threshold=0.55,
    )
    details = segmenter.segment_with_details(messages)

    segmenter.print_segments(details)

    if details.similarity_matrix:
        adjacent = [
            details.similarity_matrix[index][index + 1]
            for index in range(len(details.similarity_matrix) - 1)
        ]
        print(f"\n相邻轮次相似度: {[round(s, 4) for s in adjacent]}")
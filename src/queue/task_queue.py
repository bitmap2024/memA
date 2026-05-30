#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""通用任务队列。

适用场景：多用户、多轮对话的记忆抽取中，embedding / LLM 抽取 / 主题划分等
步骤作为独立任务异步执行。

用法::

    queue = TaskQueue(max_workers=4)
    queue.register(TaskType.EMBED, embed_handler)
    task_id = queue.submit(TaskType.EMBED, {"texts": [...]}, user_id="u1")
    task = queue.wait_one(task_id)
    queue.shutdown()
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, wait as futures_wait
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from loguru import logger


class TaskType(str, Enum):
    """任务类型，可按业务扩展。"""

    COMPRESS = "compress"
    TOPIC_SEGMENT = "topic_segment"
    LLM_EXTRACT = "llm_extract"
    EMBED = "embed"
    PERSIST = "persist"
    CUSTOM = "custom"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


Handler = Callable[["Task"], Any]


@dataclass
class Task:
    """单个异步任务。"""

    task_type: TaskType
    payload: Dict[str, Any]
    user_id: str = ""
    session_id: str = ""
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None


class TaskQueue:
    """基于线程池的简单任务队列。"""

    def __init__(self, max_workers: int = 4, name: str = "TaskQueue") -> None:
        self._name = name
        self._max_workers = max(1, int(max_workers))
        self._handlers: Dict[TaskType, Handler] = {}
        self._tasks: Dict[str, Task] = {}
        self._futures: Dict[str, Future] = {}
        self._lock = threading.Lock()
        self._executor: Optional[ThreadPoolExecutor] = None

    # ------------------------------------------------------------------
    # 注册 & 生命周期
    # ------------------------------------------------------------------
    def register(self, task_type: TaskType, handler: Handler) -> None:
        """为某种任务类型绑定处理函数。"""
        self._handlers[task_type] = handler

    def start(self) -> None:
        if self._executor is not None:
            return
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix=self._name,
        )

    def shutdown(self, wait: bool = True) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=wait)
            self._executor = None

    # ------------------------------------------------------------------
    # 提交 & 查询
    # ------------------------------------------------------------------
    def submit(
        self,
        task_type: TaskType,
        payload: Dict[str, Any],
        *,
        user_id: str = "",
        session_id: str = "",
    ) -> str:
        """提交任务，返回 task_id。"""
        self.start()
        task = Task(
            task_type=task_type,
            payload=payload,
            user_id=user_id,
            session_id=session_id,
        )
        with self._lock:
            self._tasks[task.task_id] = task
            assert self._executor is not None
            self._futures[task.task_id] = self._executor.submit(self._run, task)
        return task.task_id

    def get(self, task_id: str) -> Optional[Task]:
        with self._lock:
            return self._tasks.get(task_id)

    def wait_one(self, task_id: str, timeout: Optional[float] = None) -> Optional[Task]:
        """阻塞等待单个任务完成。"""
        with self._lock:
            future = self._futures.get(task_id)
        if future is not None:
            future.result(timeout=timeout)
        return self.get(task_id)

    def wait_all(
        self,
        task_ids: List[str],
        timeout: Optional[float] = None,
    ) -> List[Task]:
        """阻塞等待多个任务完成。"""
        with self._lock:
            futures = [self._futures[tid] for tid in task_ids if tid in self._futures]
        if futures:
            futures_wait(futures, timeout=timeout)
        return [t for tid in task_ids if (t := self.get(tid)) is not None]

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _run(self, task: Task) -> Any:
        task.status = TaskStatus.RUNNING
        handler = self._handlers.get(task.task_type)
        if handler is None:
            task.status = TaskStatus.FAILED
            task.error = f"未注册 handler: {task.task_type.value}"
            logger.warning(f"[{self._name}] {task.error} task_id={task.task_id}")
            return None
        try:
            task.result = handler(task)
            task.status = TaskStatus.DONE
            return task.result
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            logger.warning(
                f"[{self._name}] 任务失败 type={task.task_type.value} "
                f"user={task.user_id} task_id={task.task_id}: {e}"
            )
            raise


def build_memory_extract_queue(pipeline: Any, max_workers: int = 4) -> TaskQueue:
    """为 MemoryExtractPipeline 预注册常用 handler，开箱即用。

    Args:
        pipeline: MemoryExtractPipeline 实例
        max_workers: 线程池大小
    """
    queue = TaskQueue(max_workers=max_workers, name="MemoryExtract")

    queue.register(
        TaskType.COMPRESS,
        lambda t: pipeline.compress_messages(
            t.payload["messages"],
            rate=t.payload.get("rate"),
        ),
    )
    queue.register(
        TaskType.TOPIC_SEGMENT,
        lambda t: pipeline.segment_topics(t.payload["messages"]),
    )
    queue.register(
        TaskType.LLM_EXTRACT,
        lambda t: pipeline.extractor.extract_all(t.payload["topic_text"]),
    )
    queue.register(
        TaskType.EMBED,
        lambda t: pipeline.embedder.encode(t.payload["texts"]),
    )

    return queue

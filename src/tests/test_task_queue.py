#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import time

from src.queue.task_queue import TaskQueue, TaskStatus, TaskType


def test_submit_and_wait():
    queue = TaskQueue(max_workers=2)

    queue.register(
        TaskType.CUSTOM,
        lambda t: t.payload["x"] + 1,
    )

    tid = queue.submit(TaskType.CUSTOM, {"x": 1}, user_id="user_a")
    task = queue.wait_one(tid, timeout=5)

    assert task is not None
    assert task.status == TaskStatus.DONE
    assert task.result == 2
    assert task.user_id == "user_a"

    queue.shutdown()


def test_parallel_tasks():
    queue = TaskQueue(max_workers=4)

    queue.register(
        TaskType.CUSTOM,
        lambda t: (time.sleep(0.05), t.payload["n"] * 2)[1],
    )

    ids = [
        queue.submit(TaskType.CUSTOM, {"n": i}, user_id=f"user_{i}")
        for i in range(4)
    ]
    tasks = queue.wait_all(ids, timeout=5)

    assert len(tasks) == 4
    assert all(t.status == TaskStatus.DONE for t in tasks)
    assert sorted(t.result for t in tasks) == [0, 2, 4, 6]

    queue.shutdown()


def test_missing_handler():
    queue = TaskQueue(max_workers=1)
    tid = queue.submit(TaskType.EMBED, {"texts": ["hi"]})
    task = queue.wait_one(tid, timeout=5)

    assert task is not None
    assert task.status == TaskStatus.FAILED
    assert "未注册 handler" in (task.error or "")

    queue.shutdown()

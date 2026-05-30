#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""时间衰减函数：exp(-Δt / τ)。Δt 单位为天，τ 默认 30 天。"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional


def _parse_iso(time_str: Optional[str]) -> Optional[datetime]:
    if not time_str:
        return None
    try:
        dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def time_decay_factor(
    updated_at: Optional[str],
    now: Optional[datetime] = None,
    tau_days: float = 30.0,
) -> float:
    """exp(-Δt_days / tau)。无效时间返回 1.0（不衰减）。"""
    if tau_days <= 0:
        return 1.0
    dt = _parse_iso(updated_at)
    if dt is None:
        return 1.0
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    delta_days = max(0.0, (now - dt).total_seconds() / 86400.0)
    return float(math.exp(-delta_days / tau_days))

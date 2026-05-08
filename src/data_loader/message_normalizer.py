import uuid
import re
import copy
import concurrent
import logging
import json
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List, Tuple


class MessageNormalizer:

    _SESSION_RE = re.compile(
        r'(?P<date>\d{4}[/-]\d{1,2}[/-]\d{1,2})\s*\((?P<weekday>[^)]+)\)\s*(?P<time>\d{1,2}:\d{2}(?::\d{2})?)'
    )

    def __init__(self, offset_ms: int = 1000):
        self.last_timestamp_map: Dict[str, datetime] = {}
        self.offset = timedelta(milliseconds=offset_ms)

    def _parse_session_timestamp(self, raw_ts: str) -> Tuple[datetime, str]:
        """
        Parse a session-level timestamp and return (base_datetime, weekday).
        Supports formats like "2023/05/20 (Sat) 00:44" (also accepts '-' as separator, and optional seconds).
        Raises ValueError if parsing fails.
        """
        m = self._SESSION_RE.search(raw_ts)
        if m:
            date_str = m.group('date').replace('-', '/')
            time_str = m.group('time')
            weekday = m.group('weekday')
            fmt = "%Y/%m/%d %H:%M:%S" if time_str.count(':') == 2 else "%Y/%m/%d %H:%M"
            base_dt = datetime.strptime(f"{date_str} {time_str}", fmt)
            return base_dt, weekday

        try:
            dt = datetime.fromisoformat(raw_ts)
            return dt, dt.strftime("%a")
        except Exception as e:
            raise ValueError(f"{str(e)}: Failed to parse session time format: '{raw_ts}'. Expected something like '2023/05/20 (Sat) 00:44'")

    def normalize_messages(self, messages: Any) -> List[Dict[str, Any]]:
        """
        Accepts str / dict / list[dict]:
          - If str -> treated as a single user message (if 'time_stamp' is required, use dict form)
          - If dict -> single message
          - If list -> multiple messages (each must be a dict and contain 'time_stamp')
        Returns: List[Dict] (each item is a copied and enriched message)
        """
        # Normalize input into a list
        if isinstance(messages, dict):
            messages_list = [messages]
        elif isinstance(messages, list):
            messages_list = messages
        elif isinstance(messages, str):
            raise ValueError("Please provide messages as dict or list[dict], and ensure each dict contains a 'time_stamp' field (session-level).")
        else:
            raise ValueError("messages must be dict or list[dict] (or str, but not recommended).")

        enriched_list: List[Dict[str, Any]] = []

        for msg in messages_list:
            if not isinstance(msg, dict):
                raise ValueError("Each item in messages list must be a dict.")
            raw_ts = msg.get("time_stamp")
            if not raw_ts:
                raise ValueError("Each message should contain a 'time_stamp' field (e.g., '2023/05/20 (Sat) 00:44').")

            base_dt, weekday = self._parse_session_timestamp(raw_ts)

            # Maintain incrementing time based on raw_ts as session key
            last_dt = self.last_timestamp_map.get(raw_ts)
            if last_dt is None:
                new_dt = base_dt
            else:
                new_dt = last_dt + self.offset

            self.last_timestamp_map[raw_ts] = new_dt

            enriched = copy.deepcopy(msg)
            enriched["session_time"] = raw_ts
            enriched["time_stamp"] = new_dt.isoformat(timespec="milliseconds")
            enriched["weekday"] = weekday

            enriched_list.append(enriched)

        return enriched_list

if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "Hello, how are you?", "time_stamp": "2026-02-13 13:21:00"},
        {"role": "assistant", "content": "I'm fine, thank you!", "time_stamp": "2026-02-13 13:21:00"},
    ]
    print(messages)
    normalizer = MessageNormalizer()
    normalized_messages = normalizer.normalize_messages(messages)
    print(normalized_messages)
    print(json.dumps(normalized_messages, indent=4, ensure_ascii=False))
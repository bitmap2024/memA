"""雪花算法（Snowflake）唯一 ID 生成器。

64 位 ID 结构：
    1 位符号位（恒为 0） | 41 位时间戳（毫秒） | 5 位数据中心 ID | 5 位机器 ID | 12 位序列号
"""

import threading
import time


class SnowflakeError(Exception):
    """雪花算法相关异常。"""


class Snowflake:
    # 各部分占用的位数
    DATACENTER_ID_BITS = 5
    WORKER_ID_BITS = 5
    SEQUENCE_BITS = 12

    # 各部分最大值
    MAX_DATACENTER_ID = (1 << DATACENTER_ID_BITS) - 1  # 31
    MAX_WORKER_ID = (1 << WORKER_ID_BITS) - 1            # 31
    SEQUENCE_MASK = (1 << SEQUENCE_BITS) - 1             # 4095

    # 各部分左移位数
    WORKER_ID_SHIFT = SEQUENCE_BITS                                      # 12
    DATACENTER_ID_SHIFT = SEQUENCE_BITS + WORKER_ID_BITS                 # 17
    TIMESTAMP_SHIFT = SEQUENCE_BITS + WORKER_ID_BITS + DATACENTER_ID_BITS  # 22

    # 自定义纪元起点（毫秒），2024-01-01 00:00:00 UTC
    DEFAULT_EPOCH = 1704067200000

    def __init__(self, datacenter_id: int = 0, worker_id: int = 0, epoch: int = DEFAULT_EPOCH):
        if not (0 <= datacenter_id <= self.MAX_DATACENTER_ID):
            raise SnowflakeError(f"datacenter_id 必须在 0 ~ {self.MAX_DATACENTER_ID} 之间")
        if not (0 <= worker_id <= self.MAX_WORKER_ID):
            raise SnowflakeError(f"worker_id 必须在 0 ~ {self.MAX_WORKER_ID} 之间")

        self.datacenter_id = datacenter_id
        self.worker_id = worker_id
        self.epoch = epoch

        self._sequence = 0
        self._last_timestamp = -1
        self._lock = threading.Lock()

    @staticmethod
    def _current_millis() -> int:
        return int(time.time() * 1000)

    def _wait_next_millis(self, last_timestamp: int) -> int:
        timestamp = self._current_millis()
        while timestamp <= last_timestamp:
            timestamp = self._current_millis()
        return timestamp

    def next_id(self) -> int:
        """生成下一个全局唯一 ID（线程安全）。"""
        with self._lock:
            timestamp = self._current_millis()

            # 时钟回拨，拒绝生成 ID
            if timestamp < self._last_timestamp:
                raise SnowflakeError(
                    f"检测到时钟回拨，拒绝生成 ID，回拨 {self._last_timestamp - timestamp} 毫秒"
                )

            if timestamp == self._last_timestamp:
                # 同一毫秒内自增序列号
                self._sequence = (self._sequence + 1) & self.SEQUENCE_MASK
                if self._sequence == 0:
                    # 序列号溢出，等待下一毫秒
                    timestamp = self._wait_next_millis(self._last_timestamp)
            else:
                self._sequence = 0

            self._last_timestamp = timestamp

            return (
                ((timestamp - self.epoch) << self.TIMESTAMP_SHIFT)
                | (self.datacenter_id << self.DATACENTER_ID_SHIFT)
                | (self.worker_id << self.WORKER_ID_SHIFT)
                | self._sequence
            )


_default_snowflake = Snowflake()


def generate_id() -> int:
    """使用默认配置生成唯一 ID。"""
    return _default_snowflake.next_id()


if __name__ == "__main__":
    sf = Snowflake(datacenter_id=1, worker_id=1)
    for _ in range(5):
        print(sf.next_id())

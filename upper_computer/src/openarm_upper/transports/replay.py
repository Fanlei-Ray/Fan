from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import time

from ..messages import DetectionBatch, MotorHealth, TaskCommand, TaskStatus
from .base import DetectionCallback, MotorCallback, StatusCallback, Transport


class ReplayTransport(Transport):
    """Offline transport. It never sends commands to hardware."""

    def __init__(self, path: Path, loop: bool = True):
        self.path = Path(path)
        self.loop = bool(loop)
        self._items = self._load(self.path)
        self._index = 0
        self._on_detection: DetectionCallback = lambda _: None
        self._on_status: StatusCallback = lambda _: None
        self._on_motor: MotorCallback = lambda _: None
        self._status_queue: deque[tuple[int, TaskStatus]] = deque()
        self._estopped = False

    @staticmethod
    def _load(path: Path) -> tuple[DetectionBatch, ...]:
        items: list[DetectionBatch] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                items.append(DetectionBatch.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
        if not items:
            raise ValueError(f"replay file contains no messages: {path}")
        return tuple(items)

    def set_callbacks(self, on_detection, on_status, on_motor) -> None:
        self._on_detection = on_detection
        self._on_status = on_status
        self._on_motor = on_motor
        self._on_motor(
            MotorHealth(can_state="USART10 协议已导入；串口未启用（回放/只读）")
        )

    def emit_next_detection(self) -> None:
        if self._index >= len(self._items):
            if not self.loop:
                return
            self._index = 0
        item = self._items[self._index].refreshed()
        self._index += 1
        self._on_detection(item)

    def poll(self) -> None:
        now = time.monotonic_ns()
        while self._status_queue and self._status_queue[0][0] <= now:
            _, status = self._status_queue.popleft()
            if not self._estopped:
                self._on_status(status)

    def submit_task(self, command: TaskCommand) -> None:
        if self._estopped:
            self._on_status(
                TaskStatus(command.task_id, "FAILED", 0.0, error_code="ESTOP_LATCHED")
            )
            return
        now = time.monotonic_ns()
        arm = "right" if command.position_m and command.position_m[1] > 0.025 else "left"
        simulated = (
            (150, TaskStatus(command.task_id, "PLANNING", 0.15, arm, message="回放：规划中")),
            (700, TaskStatus(command.task_id, "EXECUTING", 0.55, arm, message="回放：执行中")),
            (1700, TaskStatus(command.task_id, "SUCCEEDED", 1.0, arm, message="回放：任务完成")),
        )
        for delay_ms, status in simulated:
            self._status_queue.append((now + delay_ms * 1_000_000, status))

    def cancel_task(self, task_id: str) -> None:
        self._status_queue.clear()
        self._on_status(TaskStatus(task_id, "CANCELLED", 0.0, message="用户取消"))

    def emergency_stop(self) -> None:
        self._estopped = True
        self._status_queue.clear()

    def reset_estop(self) -> None:
        self._estopped = False

    def close(self) -> None:
        self._status_queue.clear()

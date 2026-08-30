from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from ..messages import (
    DetectionBatch,
    LegacyVisionObservation,
    MotorHealth,
    TaskCommand,
    TaskStatus,
)


DetectionCallback = Callable[[DetectionBatch], None]
StatusCallback = Callable[[TaskStatus], None]
MotorCallback = Callable[[MotorHealth], None]
RawVisionCallback = Callable[[LegacyVisionObservation], None]
ConnectionCallback = Callable[[str, str], None]


class Transport(ABC):
    def set_raw_vision_callback(self, callback: RawVisionCallback) -> None:
        self._on_raw_vision = callback

    def set_connection_callback(self, callback: ConnectionCallback) -> None:
        self._on_connection = callback

    @abstractmethod
    def set_callbacks(
        self,
        on_detection: DetectionCallback,
        on_status: StatusCallback,
        on_motor: MotorCallback,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def poll(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def submit_task(self, command: TaskCommand) -> None:
        raise NotImplementedError

    @abstractmethod
    def cancel_task(self, task_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def emergency_stop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

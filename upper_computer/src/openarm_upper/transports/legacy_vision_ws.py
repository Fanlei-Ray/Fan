from __future__ import annotations

import asyncio
import json
from queue import Empty, Full, Queue
import threading
from typing import Any

from ..messages import LegacyVisionObservation, MotorHealth, TaskCommand
from .base import Transport


class LegacyVisionWebSocketTransport(Transport):
    """Observe the teammate's current WebSocket stream without robot control."""

    def __init__(
        self,
        url: str,
        *,
        reconnect_min_sec: float = 0.5,
        reconnect_max_sec: float = 5.0,
        max_message_bytes: int = 4_000_000,
        queue_size: int = 3,
    ):
        if not url.startswith(("ws://", "wss://")):
            raise ValueError("vision websocket_url must start with ws:// or wss://")
        self.url = url
        self.reconnect_min_sec = max(0.1, float(reconnect_min_sec))
        self.reconnect_max_sec = max(self.reconnect_min_sec, float(reconnect_max_sec))
        self.max_message_bytes = int(max_message_bytes)
        self._queue: Queue[tuple[str, Any]] = Queue(maxsize=max(1, int(queue_size)))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._on_detection = lambda _: None
        self._on_status = lambda _: None
        self._on_motor = lambda _: None
        self._on_raw_vision = lambda _: None
        self._on_connection = lambda _state, _message: None

    def set_callbacks(self, on_detection, on_status, on_motor) -> None:
        self._on_detection = on_detection
        self._on_status = on_status
        self._on_motor = on_motor
        self._on_motor(
            MotorHealth(can_state="视觉实时监视模式；电机与串口均未连接")
        )
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._thread_main,
                name="legacy-vision-websocket",
                daemon=True,
            )
            self._thread.start()

    def _put_latest(self, kind: str, payload: Any) -> None:
        try:
            self._queue.put_nowait((kind, payload))
            return
        except Full:
            pass
        try:
            self._queue.get_nowait()
        except Empty:
            pass
        try:
            self._queue.put_nowait((kind, payload))
        except Full:
            pass

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
        except Exception as exc:
            self._put_latest("connection", ("ERROR", f"视觉线程异常：{exc}"))

    async def _run(self) -> None:
        try:
            import websockets
        except ImportError:
            self._put_latest(
                "connection",
                ("DEPENDENCY_MISSING", "缺少 websockets；请安装 requirements-vision-client.txt"),
            )
            return

        delay = self.reconnect_min_sec
        while not self._stop.is_set():
            self._put_latest("connection", ("CONNECTING", self.url))
            try:
                async with websockets.connect(
                    self.url,
                    open_timeout=5,
                    close_timeout=2,
                    ping_interval=10,
                    ping_timeout=5,
                    max_size=self.max_message_bytes,
                ) as websocket:
                    self._put_latest("connection", ("CONNECTED", self.url))
                    delay = self.reconnect_min_sec
                    while not self._stop.is_set():
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue
                        if not isinstance(message, str):
                            self._put_latest("connection", ("BAD_MESSAGE", "忽略非 JSON 文本消息"))
                            continue
                        try:
                            raw = json.loads(message)
                            observation = LegacyVisionObservation.from_legacy_dict(raw)
                        except Exception as exc:
                            self._put_latest("connection", ("BAD_MESSAGE", f"视觉消息无效：{exc}"))
                            continue
                        self._put_latest("raw_vision", observation)
            except Exception as exc:
                if not self._stop.is_set():
                    self._put_latest(
                        "connection",
                        ("DISCONNECTED", f"{exc}; {delay:.1f}s 后重连"),
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * 2.0, self.reconnect_max_sec)

    def poll(self) -> None:
        for _ in range(20):
            try:
                kind, payload = self._queue.get_nowait()
            except Empty:
                break
            if kind == "raw_vision":
                self._on_raw_vision(payload)
            elif kind == "connection":
                self._on_connection(payload[0], payload[1])

    def submit_task(self, command: TaskCommand) -> None:
        raise RuntimeError("旧版视觉消息不可执行机械臂任务；仅允许观察")

    def cancel_task(self, task_id: str) -> None:
        return None

    def emergency_stop(self) -> None:
        # This transport has no motor/control connection.
        return None

    def close(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

from __future__ import annotations

import asyncio
import json
from queue import Empty, Full, Queue
import threading
from typing import Any

from ..messages import DetectionBatch, MotorHealth, TaskCommand, TaskStatus
from .base import Transport


class MujocoVisionWebSocketTransport(Transport):
    """Receive canonical detections from the local MuJoCo vision bridge.

    Task commands are sent only to the simulation bridge.  No message from
    this transport can reach serial/CAN hardware.
    """

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
            raise ValueError("MuJoCo websocket_url must start with ws:// or wss://")
        self.url = url
        self.reconnect_min_sec = max(0.1, float(reconnect_min_sec))
        self.reconnect_max_sec = max(self.reconnect_min_sec, float(reconnect_max_sec))
        self.max_message_bytes = int(max_message_bytes)
        self._queue: Queue[tuple[str, Any]] = Queue(maxsize=max(8, int(queue_size)))
        self._outgoing: Queue[dict[str, Any]] = Queue(maxsize=8)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._estopped = False
        self._connected = False
        self._active_task_id: str | None = None
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
            MotorHealth(
                connected=True,
                enabled=False,
                can_state="MuJoCo 仿真；真实串口/CAN 未打开",
            )
        )
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._thread_main,
                name="mujoco-vision-websocket",
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
            self._put_latest("connection", ("ERROR", f"MuJoCo 视觉线程异常：{exc}"))

    async def _run(self) -> None:
        try:
            import websockets
        except ImportError:
            self._put_latest(
                "connection",
                ("DEPENDENCY_MISSING", "缺少 websockets；请使用项目内 .venv"),
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
                    ping_timeout=20,
                    max_size=self.max_message_bytes,
                ) as websocket:
                    self._connected = True
                    self._put_latest("connection", ("CONNECTED", self.url))
                    delay = self.reconnect_min_sec
                    while not self._stop.is_set():
                        for _ in range(8):
                            try:
                                outgoing = self._outgoing.get_nowait()
                            except Empty:
                                break
                            await websocket.send(json.dumps(outgoing, ensure_ascii=False))
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=0.05)
                        except asyncio.TimeoutError:
                            continue
                        if not isinstance(message, str):
                            self._put_latest("connection", ("BAD_MESSAGE", "忽略非 JSON 文本消息"))
                            continue
                        try:
                            raw = json.loads(message)
                            kind = raw.get("type")
                            if kind == "detection_batch":
                                parsed_kind = "detection"
                                parsed = DetectionBatch.from_dict(raw.get("payload"))
                            elif kind == "task_status":
                                payload = raw.get("payload", {})
                                parsed_kind = "status"
                                parsed = TaskStatus(
                                    task_id=str(payload["task_id"]),
                                    state=str(payload["state"]),
                                    progress=float(payload.get("progress", 0.0)),
                                    selected_arm=payload.get("selected_arm"),
                                    error_code=payload.get("error_code"),
                                    message=str(payload.get("message", "")),
                                    stamp_ns=int(payload.get("stamp_ns", 0)),
                                )
                                if parsed.state in {"SUCCEEDED", "FAILED", "CANCELLED"}:
                                    self._active_task_id = None
                            else:
                                raise ValueError(f"unsupported message type: {kind!r}")
                        except Exception as exc:
                            self._put_latest("connection", ("BAD_MESSAGE", f"MuJoCo 消息无效：{exc}"))
                            continue
                        self._put_latest(parsed_kind, parsed)
            except Exception as exc:
                self._connected = False
                if not self._stop.is_set():
                    if self._active_task_id:
                        failed_task_id = self._active_task_id
                        self._active_task_id = None
                        self._put_latest(
                            "status",
                            TaskStatus(
                                failed_task_id,
                                "FAILED",
                                0.0,
                                error_code="SIM_BRIDGE_CONNECTION_LOST",
                                message="MuJoCo bridge 在动作途中断开，任务已终止；请查看 bridge 控制台",
                            ),
                        )
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
            if kind == "detection":
                self._on_detection(payload)
            elif kind == "status":
                self._on_status(payload)
            elif kind == "connection":
                self._on_connection(payload[0], payload[1])

    def _send_or_fail(self, envelope: dict[str, Any], task_id: str) -> None:
        if not self._connected:
            self._active_task_id = None
            self._on_status(TaskStatus(
                task_id,
                "FAILED",
                0.0,
                error_code="SIM_BRIDGE_DISCONNECTED",
                message="MuJoCo bridge 未连接",
            ))
            return
        try:
            self._outgoing.put_nowait(envelope)
        except Full:
            self._active_task_id = None
            self._on_status(TaskStatus(
                task_id,
                "FAILED",
                0.0,
                error_code="COMMAND_QUEUE_FULL",
                message="MuJoCo 命令队列已满",
            ))

    def submit_task(self, command: TaskCommand) -> None:
        if self._estopped:
            self._on_status(
                TaskStatus(command.task_id, "FAILED", 0.0, error_code="ESTOP_LATCHED")
            )
            return
        self._active_task_id = command.task_id
        self._send_or_fail(
            {"type": "task_command", "payload": command.to_dict()},
            command.task_id,
        )

    def cancel_task(self, task_id: str) -> None:
        self._send_or_fail(
            {"type": "cancel_task", "payload": {"task_id": task_id}},
            task_id,
        )

    def emergency_stop(self) -> None:
        self._estopped = True
        if self._connected:
            try:
                self._outgoing.put_nowait(
                    {"type": "estop", "payload": {"task_id": "sim-estop"}}
                )
            except Full:
                pass

    def reset_estop(self) -> None:
        self._estopped = False

    def close(self) -> None:
        self._stop.set()
        self._connected = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

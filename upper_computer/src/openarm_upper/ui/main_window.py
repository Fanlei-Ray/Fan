from __future__ import annotations

from datetime import datetime
import time
import tkinter as tk
from tkinter import messagebox, ttk
import uuid
from typing import Any

from ..messages import (
    Detection,
    DetectionBatch,
    LegacyVisionObservation,
    MotorHealth,
    TaskCommand,
    TaskStatus,
)
from ..safety import SafetyValidator, ValidationResult
from ..session_logger import SessionLogger
from ..state_machine import InvalidTransition, TaskStateMachine
from ..transports.base import Transport
from .detection_canvas import DetectionCanvas


class MainWindow:
    def __init__(
        self,
        root: tk.Tk,
        config: dict[str, Any],
        transport: Transport,
        logger: SessionLogger,
    ):
        self.root = root
        self.config = config
        self.transport = transport
        self.logger = logger
        self.validator = SafetyValidator(config)
        self.machine = TaskStateMachine()
        self.current_batch: DetectionBatch | None = None
        self.current_detection: Detection | None = None
        self.current_detections_by_id: dict[str, Detection] = {}
        self.target_choice_map: dict[str, str] = {}
        self.current_validation = ValidationResult(False, ("尚未收到检测结果",))
        self.active_task_id: str | None = None
        self._next_replay_ns = 0
        self._last_raw_ns = 0
        self._last_raw_log_ns = 0
        self._last_raw_signature = None
        self._vision_stale_reported = False

        root.title(config["app"]["title"])
        root.geometry("1280x800")
        root.minsize(1100, 700)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._configure_style()
        self._build()
        transport.set_raw_vision_callback(self.on_raw_vision)
        transport.set_connection_callback(self.on_connection)
        transport.set_callbacks(self.on_detection, self.on_status, self.on_motor)
        self.logger.log("application_started", {"config": config["_config_path"]})
        if config["transport"]["mode"] == "legacy_vision_ws":
            self._append_log("系统", "实时视觉观察模式：未标定，禁止真机运动")
        elif config["transport"]["mode"] == "mujoco_vision_ws":
            self._append_log("系统", "MuJoCo V18.3 视觉抓取联调：仅仿真，不连接真机")
        else:
            self._append_log("系统", "上位机已启动：回放模式，禁止真机运动")
        self._update_controls()
        self.root.after(50, self._poll)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Header.TLabel", font=("Microsoft YaHei UI", 18, "bold"))
        style.configure("Sub.TLabel", foreground="#475569")
        style.configure("State.TLabel", font=("Consolas", 14, "bold"))
        style.configure("Danger.TButton", font=("Microsoft YaHei UI", 11, "bold"))

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 10))
        ttk.Label(header, text="OpenArm 视觉联调上位机", style="Header.TLabel").pack(side="left")
        mode = self.config["transport"]["mode"]
        mode_text = {
            "legacy_vision_ws": "● LIVE VISION / OBSERVE ONLY",
            "mujoco_vision_ws": "● MUJOCO VISION / SIMULATION",
        }.get(mode, "● REPLAY / DRY-RUN")
        self.mode_var = tk.StringVar(value=mode_text)
        ttk.Label(header, textvariable=self.mode_var, foreground="#b45309", font=("Consolas", 12, "bold")).pack(side="right")

        body = ttk.PanedWindow(outer, orient="horizontal")
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        right = ttk.Frame(body, padding=(10, 0, 0, 0))
        body.add(left, weight=3)
        body.add(right, weight=2)

        self.canvas = DetectionCanvas(left)
        self.canvas.pack(fill="both", expand=True)
        self.image_note = tk.StringVar(value="等待检测消息…")
        ttk.Label(left, textvariable=self.image_note, style="Sub.TLabel").pack(anchor="w", pady=(5, 0))

        status_box = ttk.LabelFrame(right, text="系统与任务状态", padding=10)
        status_box.pack(fill="x")
        self.state_var = tk.StringVar(value="IDLE")
        if mode == "legacy_vision_ws":
            initial_connection = "视觉 WebSocket：等待连接 / ROS 2：待接入"
        elif mode == "mujoco_vision_ws":
            initial_connection = "MuJoCo 视觉桥：等待连接 / 真机：未连接"
        else:
            initial_connection = "视觉回放：已配置 / ROS 2：待接入"
        self.connection_var = tk.StringVar(value=initial_connection)
        self.motor_var = tk.StringVar(value="电机/CAN：待接入")
        ttk.Label(status_box, textvariable=self.state_var, style="State.TLabel").pack(anchor="w")
        ttk.Label(status_box, textvariable=self.connection_var).pack(anchor="w", pady=(5, 0))
        ttk.Label(status_box, textvariable=self.motor_var).pack(anchor="w")

        target_box = ttk.LabelFrame(right, text="目标选择与当前目标", padding=10)
        target_box.pack(fill="x", pady=10)
        self.target_choice_var = tk.StringVar(value="等待识别目标…")
        self.target_choice = ttk.Combobox(
            target_box,
            textvariable=self.target_choice_var,
            state="readonly",
            values=(),
        )
        ttk.Label(target_box, text="指定抓取：", width=11).grid(row=0, column=0, sticky="nw", pady=2)
        self.target_choice.grid(row=0, column=1, sticky="ew", pady=2)
        self.target_choice.bind("<<ComboboxSelected>>", self._on_target_choice)
        self.target_vars = {name: tk.StringVar(value="—") for name in ("id", "class", "confidence", "source", "frame", "position", "safety")}
        labels = (("目标 ID", "id"), ("类别", "class"), ("置信度", "confidence"), ("检测来源", "source"), ("坐标系", "frame"), ("位置 (m)", "position"), ("安全检查", "safety"))
        for row, (label, key) in enumerate(labels, start=1):
            ttk.Label(target_box, text=label + "：", width=11).grid(row=row, column=0, sticky="nw", pady=2)
            ttk.Label(target_box, textvariable=self.target_vars[key], wraplength=360).grid(row=row, column=1, sticky="nw", pady=2)
        target_box.columnconfigure(1, weight=1)

        controls_title = (
            "操作（仅驱动 MuJoCo，绝不连接真机）"
            if mode == "mujoco_vision_ws"
            else "操作（当前均为 dry-run）"
        )
        controls = ttk.LabelFrame(right, text=controls_title, padding=10)
        controls.pack(fill="x")
        self.confirm_button = ttk.Button(controls, text="确认并执行", command=self.confirm_task)
        self.cancel_button = ttk.Button(controls, text="取消当前任务", command=self.cancel_task)
        self.home_button = ttk.Button(controls, text="回零（模拟）", command=self.home_task)
        self.shuffle_button = ttk.Button(
            controls,
            text="重新打乱物品",
            command=self.shuffle_scene,
        )
        self.estop_button = tk.Button(controls, text="急 停", command=self.emergency_stop, bg="#dc2626", fg="white", activebackground="#991b1b", activeforeground="white", font=("Microsoft YaHei UI", 13, "bold"), relief="raised")
        self.reset_button = ttk.Button(controls, text="复位急停", command=self.reset_estop)
        self.confirm_button.grid(row=0, column=0, sticky="ew", padx=3, pady=3)
        self.cancel_button.grid(row=0, column=1, sticky="ew", padx=3, pady=3)
        self.home_button.grid(row=1, column=0, sticky="ew", padx=3, pady=3)
        self.shuffle_button.grid(row=1, column=1, sticky="ew", padx=3, pady=3)
        self.reset_button.grid(row=2, column=0, columnspan=2, sticky="ew", padx=3, pady=3)
        self.estop_button.grid(row=3, column=0, columnspan=2, sticky="ew", padx=3, pady=(8, 3), ipady=6)
        controls.columnconfigure((0, 1), weight=1)

        log_box = ttk.LabelFrame(right, text="联调日志", padding=6)
        log_box.pack(fill="both", expand=True, pady=(10, 0))
        self.log_text = tk.Text(log_box, height=12, wrap="word", state="disabled", bg="#f8fafc", font=("Consolas", 9))
        scrollbar = ttk.Scrollbar(log_box, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _append_log(self, source: str, message: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {source}: {message}\n"
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_state(self, state: str) -> None:
        try:
            self.machine.transition(state)
        except InvalidTransition as exc:
            self._append_log("状态机", str(exc))
            self.logger.log("invalid_transition", {"from": self.machine.state, "to": state, "error": str(exc)})
            return
        self.state_var.set(self.machine.state)
        self._update_controls()

    def _update_controls(self) -> None:
        ready = self.machine.state == "TARGET_READY" and self.current_validation.accepted
        self.confirm_button.configure(state="normal" if ready else "disabled")
        active = self.machine.state in {"CONFIRMED", "PLANNING", "EXECUTING"}
        self.cancel_button.configure(state="normal" if active else "disabled")
        idle = self.machine.state in {"IDLE", "TARGET_READY", "SUCCEEDED", "FAILED", "CANCELLED"}
        self.home_button.configure(state="normal" if idle and not self.machine.estop_latched else "disabled")
        shuffle_supported = bool(
            self.config["transport"]["mode"] == "mujoco_vision_ws"
            and self.config.get("simulation", {}).get("allow_safe_shuffle", False)
        )
        self.shuffle_button.configure(
            state=(
                "normal"
                if idle and shuffle_supported and not self.machine.estop_latched
                else "disabled"
            )
        )
        self.reset_button.configure(state="normal" if self.machine.estop_latched else "disabled")

    def on_detection(self, batch: DetectionBatch) -> None:
        if self.config["transport"]["mode"] == "mujoco_vision_ws":
            self._last_raw_ns = time.time_ns()
            self._vision_stale_reported = False
        self.current_batch = batch
        self.canvas.draw(batch)
        age_ms = (time.time_ns() - batch.stamp_ns) / 1_000_000
        self.image_note.set(f"{batch.frame_id} · {batch.image_size[0]}×{batch.image_size[1]} · age {age_ms:.0f} ms · {len(batch.detections)} 个目标")
        if not batch.detections:
            self.current_detections_by_id.clear()
            self.target_choice_map.clear()
            self.target_choice.configure(values=())
            self.target_choice_var.set("没有识别目标")
            self.canvas.set_selected_id(None)
            self.current_detection = None
            self.current_validation = ValidationResult(False, ("没有检测目标",))
            self._show_target(None)
            if self.machine.state not in {"PLANNING", "EXECUTING", "ESTOP"}:
                self._set_state("IDLE")
            return

        self.current_detections_by_id = {item.id: item for item in batch.detections}
        previous_id = self.current_detection.id if self.current_detection else None
        labels = []
        self.target_choice_map.clear()
        for item in sorted(batch.detections, key=lambda value: (value.class_name, value.id)):
            source = item.source or "未声明来源"
            label = f"{item.class_name} | {item.id} | {source}"
            labels.append(label)
            self.target_choice_map[label] = item.id
        self.target_choice.configure(values=labels)
        selected = self.current_detections_by_id.get(previous_id or "")
        if selected is None:
            selected = max(batch.detections, key=lambda item: item.confidence)
        selected_label = next(
            label for label, detection_id in self.target_choice_map.items()
            if detection_id == selected.id
        )
        self.target_choice_var.set(selected_label)
        self._apply_selected_detection(selected)

    def _on_target_choice(self, _event: Any = None) -> None:
        detection_id = self.target_choice_map.get(self.target_choice_var.get())
        selected = self.current_detections_by_id.get(detection_id or "")
        if selected is None or self.current_batch is None:
            return
        self._apply_selected_detection(selected)
        self._append_log("目标选择", f"指定抓取 {selected.class_name}（{selected.id}）")

    def _apply_selected_detection(self, selected: Detection) -> None:
        if self.current_batch is None:
            return
        validation = self.validator.validate(self.current_batch, selected)
        self.current_detection = selected
        self.current_validation = validation
        self.canvas.set_selected_id(selected.id)
        self._show_target(selected)
        self.logger.log("detection_received", {"batch": {"frame_id": self.current_batch.frame_id, "stamp_ns": self.current_batch.stamp_ns}, "detection": selected, "accepted": validation.accepted, "reasons": validation.reasons})
        if self.machine.state in {"IDLE", "TARGET_READY", "SUCCEEDED", "FAILED", "CANCELLED"}:
            if self.machine.state != "IDLE":
                self._set_state("IDLE")
            if validation.accepted:
                self._set_state("TARGET_READY")
                self._append_log("视觉", f"目标可执行：{selected.id}")
            else:
                self._append_log("安全", "；".join(validation.reasons))

    def on_raw_vision(self, observation: LegacyVisionObservation) -> None:
        """Display legacy pixels/depth while deliberately blocking execution."""
        self._last_raw_ns = observation.received_ns
        self._vision_stale_reported = False
        self.current_batch = None
        self.current_detection = None
        self.current_validation = ValidationResult(
            False,
            ("仅有像素中心和轴向深度；缺少三维反投影及 base_link 标定",),
        )
        self.canvas.draw_raw(observation)
        age_ms = (time.time_ns() - observation.received_ns) / 1_000_000
        self.image_note.set(
            f"视觉同学旧版流 · 640×480 · age {age_ms:.0f} ms · "
            "像素+深度（不可执行）"
        )
        if observation.detected:
            u, v = observation.pixel_center_uv
            self.target_vars["id"].set("旧协议未提供")
            self.target_vars["class"].set(observation.class_name or "unknown")
            self.target_vars["confidence"].set("旧协议未提供")
            self.target_vars["frame"].set("camera pixel/depth，未标定")
            self.target_vars["position"].set(
                f"u={u}px, v={v}px, depth={observation.depth_m:.3f}m"
            )
            self.target_vars["safety"].set(
                "拒绝：不是 base_link 三维坐标，缺少 bbox/置信度/时间戳"
            )
        else:
            self._show_target(None)
            self.target_vars["safety"].set("无目标；仅观察")

        if self.machine.state in {"TARGET_READY", "SUCCEEDED", "FAILED", "CANCELLED"}:
            self._set_state("IDLE")
        self._update_controls()

        signature = (
            observation.detected,
            observation.class_name,
            observation.pixel_center_uv,
            round(observation.depth_m, 3),
        )
        now = time.time_ns()
        if signature != self._last_raw_signature or now - self._last_raw_log_ns >= 1_000_000_000:
            self.logger.log("legacy_vision_observation", observation.to_log_dict())
            self._last_raw_signature = signature
            self._last_raw_log_ns = now

    def on_connection(self, state: str, message: str) -> None:
        connected_label = (
            "已连接（MuJoCo 仿真）"
            if self.config["transport"]["mode"] == "mujoco_vision_ws"
            else "已连接（仅观察）"
        )
        labels = {
            "CONNECTING": "连接中",
            "CONNECTED": connected_label,
            "DISCONNECTED": "已断开/重连中",
            "DEPENDENCY_MISSING": "依赖缺失",
            "BAD_MESSAGE": "消息异常",
            "ERROR": "线程异常",
        }
        self.connection_var.set(f"视觉 WebSocket：{labels.get(state, state)} · {message}")
        if state != "CONNECTED":
            self._append_log("视觉连接", f"{state}: {message}")
        self.logger.log("vision_connection", {"state": state, "message": message})

    def _show_target(self, detection: Detection | None) -> None:
        if detection is None or self.current_batch is None:
            for variable in self.target_vars.values():
                variable.set("—")
            return
        x, y, z = detection.position_m
        self.target_vars["id"].set(detection.id)
        self.target_vars["class"].set(detection.class_name)
        self.target_vars["confidence"].set(f"{detection.confidence:.3f}")
        self.target_vars["source"].set(detection.source or "未声明")
        self.target_vars["frame"].set(self.current_batch.frame_id)
        self.target_vars["position"].set(f"x={x:.3f}, y={y:.3f}, z={z:.3f}")
        self.target_vars["safety"].set("通过" if self.current_validation.accepted else "拒绝：" + "；".join(self.current_validation.reasons))

    def confirm_task(self) -> None:
        if not (self.current_batch and self.current_detection and self.current_validation.accepted):
            return
        self._set_state("CONFIRMED")
        task_id = f"pick-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
        self.active_task_id = task_id
        command = TaskCommand(
            task_id=task_id,
            command="pick_place",
            object_id=self.current_detection.id,
            frame_id=self.current_batch.frame_id,
            position_m=self.current_detection.position_m,
            orientation_xyzw=self.current_detection.orientation_xyzw,
            object_class=self.current_detection.class_name,
            requested_arm=str(
                self.config.get("simulation", {}).get("preferred_arm", "auto")
            ),
            dry_run=bool(self.config["app"]["dry_run"]),
        )
        self.logger.log("task_submitted", command)
        scope = (
            "MuJoCo V18.3 仿真执行"
            if self.config["transport"]["mode"] == "mujoco_vision_ws"
            else "dry-run"
        )
        self._append_log("任务", f"已确认 {task_id}（{scope}）")
        self.transport.submit_task(command)

    def home_task(self) -> None:
        if self.machine.state != "IDLE":
            self._set_state("IDLE")
        task_id = f"home-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.active_task_id = task_id
        simulation = self.config["transport"]["mode"] == "mujoco_vision_ws"
        command = TaskCommand(
            task_id,
            "home",
            None,
            "base_link",
            None,
            None,
            dry_run=not simulation,
        )
        self.logger.log("home_submitted", command)
        self._append_log(
            "任务",
            f"回零请求 {task_id}（{'MuJoCo home keyframe' if simulation else '仅模拟'}）",
        )
        self.transport.submit_task(command)

    def shuffle_scene(self) -> None:
        if not self.config.get("simulation", {}).get("allow_safe_shuffle", False):
            return
        if self.machine.state != "IDLE":
            self._set_state("IDLE")
        task_id = f"shuffle-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.active_task_id = task_id
        command = TaskCommand(
            task_id,
            "shuffle_scene",
            None,
            "base_link",
            None,
            None,
            dry_run=False,
        )
        self.logger.log("shuffle_submitted", command)
        self._append_log("场景", f"请求重新打乱物品：{task_id}")
        self.transport.submit_task(command)

    def cancel_task(self) -> None:
        if not self.active_task_id:
            return
        self.transport.cancel_task(self.active_task_id)
        self.logger.log("cancel_requested", {"task_id": self.active_task_id})

    def emergency_stop(self) -> None:
        self.transport.emergency_stop()
        self.machine.emergency_stop()
        self.state_var.set("ESTOP（已锁存）")
        self._append_log("急停", "软件急停已锁存；真机仍必须使用物理急停")
        self.logger.log("estop_latched")
        self._update_controls()

    def reset_estop(self) -> None:
        if not messagebox.askyesno("复位急停", "确认现场安全、硬件急停已复位且故障原因已排除？"):
            return
        self.machine.reset_estop()
        reset = getattr(self.transport, "reset_estop", None)
        if callable(reset):
            reset()
        self.state_var.set("IDLE")
        self._append_log("急停", "人工复位完成；仍处于 dry-run")
        self.logger.log("estop_reset")
        self._update_controls()

    def on_status(self, status: TaskStatus) -> None:
        if self.active_task_id and status.task_id != self.active_task_id:
            self._append_log("状态", f"忽略其他任务状态：{status.task_id}")
            return
        self.active_task_id = status.task_id
        self._set_state(status.state)
        arm = f" arm={status.selected_arm}" if status.selected_arm else ""
        self._append_log("状态", f"{status.state} {status.progress:.0%}{arm} {status.message}")
        self.logger.log("task_status", status)

    def on_motor(self, health: MotorHealth) -> None:
        self.motor_var.set(f"电机/CAN：{health.can_state} · enabled={health.enabled}")
        self.logger.log("motor_health", health)

    def _poll(self) -> None:
        self.transport.poll()
        now = time.monotonic_ns()
        if self.config["transport"]["mode"] == "replay" and now >= self._next_replay_ns:
            emitter = getattr(self.transport, "emit_next_detection", None)
            if callable(emitter):
                emitter()
            interval_ms = int(self.config["transport"]["interval_ms"])
            self._next_replay_ns = now + interval_ms * 1_000_000
        if (
            self.config["transport"]["mode"] in {"legacy_vision_ws", "mujoco_vision_ws"}
            and self._last_raw_ns
            and time.time_ns() - self._last_raw_ns > 2_500_000_000
            and not self._vision_stale_reported
            and self.machine.state not in {"PLANNING", "EXECUTING"}
        ):
            self._vision_stale_reported = True
            self.connection_var.set("视觉 WebSocket：消息超时（>2.5s），禁止执行")
            self._append_log("视觉安全", "实时消息已超时")
            self.logger.log("vision_stale")
        self.root.after(50, self._poll)

    def close(self) -> None:
        self.logger.log("application_closed")
        self.transport.close()
        self.root.destroy()

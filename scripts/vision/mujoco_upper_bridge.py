from __future__ import annotations

"""Stream MuJoCo camera detections to the OpenArm upper computer.

This process is a simulation-only bridge. It renders the overhead MuJoCo
camera, runs YOLOv8 with an explicit rendered-RGB fallback, and publishes a
multi-object DetectionBatch over WebSocket. It never opens serial/CAN devices.
"""

import argparse
import asyncio
import base64
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import queue
import sys
import time
from typing import Any

import mujoco
import numpy as np
from PIL import Image
import websockets


VISION_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = VISION_DIR.parent
ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from vision.vision_cube_detector import CAMERA_NAME, ensure_vision_xml  # noqa: E402
from vision.multi_object_detector import (  # noqa: E402
    DEFAULT_WEIGHTS,
    MultiObjectDetector,
    SimulationObjectSpec,
)
import task_planner.core as core  # noqa: E402


SOURCE_XML = ROOT / "v2" / "demo_multi_object.xml"
VISION_XML = ROOT / "v2" / "demo_multi_object_vision.xml"
BRIDGE_OUTPUT_DIR = ROOT / "outputs" / "upper_computer_mujoco_bridge"
INITIAL_OBJECT_SPAWNS = {
    # Project V18's recorded successful right-arm pick point.  Keep this as
    # the simulation loading station instead of replacing it with a noisy
    # image-to-world estimate (an 18 mm x error was enough to push the part).
    "orange_cube": np.array([0.516, 0.050, 1.050], dtype=float),
    # The remaining items are kept in a separate display row so the open
    # gripper cannot sweep an unselected neighbour while approaching the slot.
    "blue_part": np.array([0.440, -0.075, 1.050], dtype=float),
    "phone": np.array([0.520, -0.075, 1.050], dtype=float),
    "mouse": np.array([0.600, -0.075, 1.050], dtype=float),
}
DEFAULT_CUBE_SPAWN = INITIAL_OBJECT_SPAWNS["orange_cube"].copy()
SAFE_PICK_X = (0.42, 0.66)
SAFE_PICK_Y = (-0.12, 0.20)


def prepare_multi_object_scene(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    viewer: Any = None,
    realtime: bool = False,
) -> dict[str, list[float]]:
    """Restore the proven home posture and all simulation objects."""
    core.load_home(model, data)
    for body_name, position in INITIAL_OBJECT_SPAWNS.items():
        core.set_free_body_pos(model, data, body_name, position)
    for actuator, value in (
        ("left_finger1_ctrl", core.get_left_attr("LEFT_FINGER_PRE_OPEN", 0.445)),
        ("right_finger1_ctrl", core.right_rule.RIGHT_FINGER_PRE_OPEN),
        ("lifter_ctrl", 0.0),
    ):
        if core.maybe_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator) != -1:
            core.set_ctrl(model, data, actuator, value)
    mujoco.mj_forward(model, data)
    core.sim_steps(model, data, 360, viewer=viewer, realtime=realtime)
    return {
        name: [float(value) for value in core.get_body_pos(model, data, name)]
        for name in INITIAL_OBJECT_SPAWNS
    }


class PlannerViewerProxy:
    """Capture state snapshots from the same thread that advances MuJoCo.

    The existing V18.3 motion code calls ``viewer.sync()`` after each physics
    step. Hooking that call lets the planner thread copy a consistent state;
    the asyncio thread renders that snapshot through a separate MjData.
    """

    handles_realtime_pacing = True

    def __init__(
        self,
        viewer: Any | None,
        capture_callback: Any,
        pace_callback: Any | None = None,
    ):
        # ``viewer`` is accepted for compatibility with the existing planner
        # call sites, but it must never be touched from the planner worker.
        # MuJoCo's passive viewer is owned exclusively by the main/UI thread.
        self._viewer = viewer
        self._capture_callback = capture_callback
        self._pace_callback = pace_callback

    def is_running(self) -> bool:
        # A transient viewer failure must not be interpreted as "finish the
        # robot motion now".  That previously left the arm suspended halfway.
        return True

    def sync(self) -> None:
        # The motion code calls sync after each mj_step.  Only copy a snapshot
        # here; rendering and viewer.sync happen on the asyncio/main thread.
        self._capture_callback()
        if self._pace_callback is not None:
            self._pace_callback()


@dataclass(frozen=True)
class SimulationSnapshot:
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    act: np.ndarray
    mocap_pos: np.ndarray
    mocap_quat: np.ndarray
    time: float


def patch_bridge_output_paths() -> None:
    """Keep bridge runs separate from the user's existing V18.3 evidence."""
    core.OUTPUT_DIR = BRIDGE_OUTPUT_DIR
    core.LOG_PATH = BRIDGE_OUTPUT_DIR / "task_log.csv"
    core.PLAN_PATH = BRIDGE_OUTPUT_DIR / "execution_plan.csv"
    core.SUMMARY_PATH = BRIDGE_OUTPUT_DIR / "task_summary.json"
    core.REPORT_PATH = BRIDGE_OUTPUT_DIR / "presentation_report.md"
    core.STATE_MACHINE_PATH = BRIDGE_OUTPUT_DIR / "state_machine.mmd"
    core.RUNBOOK_PATH = BRIDGE_OUTPUT_DIR / "demo_runbook.txt"
    core.PATH_PLAN_PATH = BRIDGE_OUTPUT_DIR / "path_plan.csv"
    core.COLLISION_LOG_PATH = BRIDGE_OUTPUT_DIR / "collision_log.csv"


class MujocoUpperBridge:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        fps: float,
        *,
        viewer: Any = None,
        viewer_data: mujoco.MjData | None = None,
        realtime: bool = False,
        detector_backend: str = "auto",
        yolo_weights: str | Path = DEFAULT_WEIGHTS,
        yolo_confidence: float = 0.35,
        direct_pose_execution: bool = False,
        object_rule_configs: dict[str, dict[str, Any]] | None = None,
        rule_speed_scale: float = 1.0,
        detector_specs: tuple[SimulationObjectSpec, ...] | None = None,
        realtime_playback_rate: float = 1.0,
    ):
        self.model = model
        self.data = data
        self.viewer = viewer
        self.viewer_data = viewer_data
        self.realtime = bool(realtime)
        self.realtime_playback_rate = max(0.25, float(realtime_playback_rate))
        self._pace_sim_origin: float | None = None
        self._pace_wall_origin = 0.0
        self.fps = max(1.0, min(float(fps), 20.0))
        # Rendering owns a different MjData from the planner. This is essential:
        # the OpenGL renderer stays on the asyncio thread while mj_step runs in
        # the worker thread.
        self.render_data = mujoco.MjData(model)
        self.detector = MultiObjectDetector(
            model,
            self.render_data,
            backend=detector_backend,
            weights=yolo_weights,
            confidence=yolo_confidence,
            **({} if detector_specs is None else {"specs": detector_specs}),
        )
        self.clients: set[Any] = set()
        self.command_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=4)
        self.last_message = ""
        self.frame_index = 0
        self.last_planner_cube_pos = DEFAULT_CUBE_SPAWN.copy()
        self.last_detections_by_id: dict[str, Any] = {}
        self.last_execution_metrics: dict[str, Any] = {}
        self.pick_slot_object = "orange_cube"
        self.direct_pose_execution = bool(direct_pose_execution)
        self.object_rule_configs = dict(object_rule_configs or {})
        self.rule_speed_scale = float(rule_speed_scale)
        self.active_command_task: asyncio.Task[None] | None = None
        self.active_task_id: str | None = None
        self.active_task_started_at = 0.0
        self.last_progress_status_at = 0.0
        self.execution_status_started = False
        self.live_states: queue.Queue[SimulationSnapshot] = queue.Queue(maxsize=1)
        self.last_live_capture_at = 0.0
        self.planner_viewer = PlannerViewerProxy(
            viewer,
            self.capture_live_state,
            self.pace_live_motion,
        )

    @property
    def command_active(self) -> bool:
        return bool(
            self.active_command_task is not None
            and not self.active_command_task.done()
        )

    def set_viewer(self, viewer: Any, viewer_data: mujoco.MjData | None = None) -> None:
        self.viewer = viewer
        if viewer_data is not None:
            self.viewer_data = viewer_data
        self.planner_viewer = PlannerViewerProxy(
            viewer,
            self.capture_live_state,
            self.pace_live_motion,
        )

    def reset_realtime_pacing(self) -> None:
        self._pace_sim_origin = None
        self._pace_wall_origin = 0.0

    def pace_live_motion(self) -> None:
        """Pace batches of physics steps without Windows 2 ms sleep inflation."""
        if not self.realtime or self.viewer is None:
            return
        sim_now = float(self.data.time)
        wall_now = time.perf_counter()
        if self._pace_sim_origin is None:
            self._pace_sim_origin = sim_now
            self._pace_wall_origin = wall_now
            return
        desired_wall_elapsed = (
            sim_now - self._pace_sim_origin
        ) / self.realtime_playback_rate
        actual_wall_elapsed = wall_now - self._pace_wall_origin
        ahead = desired_wall_elapsed - actual_wall_elapsed
        # Sleeping once per 8-12 ms batch avoids thousands of inaccurate
        # 2 ms sleeps on Python 3.10/Windows while keeping motion smooth.
        if ahead >= 0.010:
            time.sleep(max(0.0, ahead - 0.002))

    async def handler(self, websocket: Any) -> None:
        self.clients.add(websocket)
        print(f"[bridge] client connected: {websocket.remote_address}")
        try:
            if self.last_message:
                await websocket.send(self.last_message)
            async for message in websocket:
                try:
                    decoded = json.loads(message)
                    if decoded.get("type") not in {"task_command", "cancel_task", "estop"}:
                        raise ValueError("unsupported message type")
                    self.command_queue.put_nowait(decoded)
                    print(f"[bridge] queued {decoded.get('type')!r}")
                except asyncio.QueueFull:
                    await self.send_status(
                        str(decoded.get("payload", {}).get("task_id", "unknown")),
                        "FAILED",
                        0.0,
                        error_code="COMMAND_QUEUE_FULL",
                        message="MuJoCo task queue is full",
                    )
                except Exception as exc:
                    print(f"[bridge] ignored invalid inbound message: {exc}")
        except Exception as exc:
            # Client windows may be closed without a WebSocket close frame on
            # Windows.  This is a client disconnect, not a bridge crash.
            print(f"[bridge] client connection ended: {exc}")
        finally:
            self.clients.discard(websocket)
            print("[bridge] client disconnected")

    async def broadcast(self, message: str) -> None:
        if not self.clients:
            return
        clients = tuple(self.clients)
        results = await asyncio.gather(
            *(client.send(message) for client in clients),
            return_exceptions=True,
        )
        for client, result in zip(clients, results):
            if isinstance(result, Exception):
                self.clients.discard(client)

    async def send_status(
        self,
        task_id: str,
        state: str,
        progress: float,
        *,
        selected_arm: str | None = None,
        error_code: str | None = None,
        message: str = "",
    ) -> None:
        envelope = {
            "type": "task_status",
            "payload": {
                "task_id": task_id,
                "state": state,
                "progress": float(progress),
                "selected_arm": selected_arm,
                "error_code": error_code,
                "message": message,
                "stamp_ns": time.time_ns(),
            },
        }
        await self.broadcast(json.dumps(envelope, ensure_ascii=False))

    async def process_one_command(self) -> None:
        if self.active_command_task is not None:
            if not self.active_command_task.done():
                return
            # Retrieve any unexpected task exception so asyncio does not hide it.
            try:
                self.active_command_task.result()
            except Exception as exc:
                print(f"[bridge] command task failed unexpectedly: {exc}")
            self.active_command_task = None

        try:
            envelope = self.command_queue.get_nowait()
        except asyncio.QueueEmpty:
            return

        self.active_command_task = asyncio.create_task(
            self.execute_command_guarded(envelope),
            name=f"mujoco-command-{time.time_ns()}",
        )

    async def execute_command_guarded(self, envelope: dict[str, Any]) -> None:
        """Guarantee a terminal failure status for every unexpected error."""
        payload = envelope.get("payload", {})
        task_id = str(payload.get("task_id", f"sim-{time.time_ns()}"))
        self.active_task_id = task_id
        self.active_task_started_at = time.monotonic()
        self.last_progress_status_at = 0.0
        self.execution_status_started = False
        self.reset_realtime_pacing()
        try:
            await self.execute_command(envelope)
        except Exception as exc:
            print(f"[bridge] command {task_id} failed unexpectedly: {exc}")
            await self.send_status(
                task_id,
                "FAILED",
                0.0,
                error_code="SIM_BRIDGE_INTERNAL_ERROR",
                message=f"MuJoCo bridge 内部错误：{exc}",
            )
        finally:
            self.active_task_id = None
            self.execution_status_started = False

    async def publish_progress_heartbeat(self) -> None:
        """Keep the upper computer responsive during a blocking motion."""
        if (
            not self.command_active
            or not self.active_task_id
            or not self.execution_status_started
        ):
            return
        now = time.monotonic()
        if now - self.last_progress_status_at < 1.5:
            return
        self.last_progress_status_at = now
        elapsed = max(0.0, now - self.active_task_started_at)
        progress = min(0.92, 0.45 + elapsed * 0.035)
        await self.send_status(
            self.active_task_id,
            "EXECUTING",
            progress,
            selected_arm="right",
            message=f"实时执行中（{elapsed:.1f}s），画面与物理状态持续更新",
        )

    async def execute_command(self, envelope: dict[str, Any]) -> None:

        kind = str(envelope.get("type", ""))
        payload = envelope.get("payload", {})
        task_id = str(payload.get("task_id", f"sim-{time.time_ns()}"))
        if kind in {"cancel_task", "estop"}:
            await self.send_status(
                task_id,
                "CANCELLED" if kind == "cancel_task" else "FAILED",
                0.0,
                error_code=None if kind == "cancel_task" else "SIM_ESTOP",
                message="请求已记录；现有 V18.3 阻塞式动作不能中途打断",
            )
            return

        command = str(payload.get("command", ""))
        if command == "home":
            await self.send_status(task_id, "PLANNING", 0.2, message="加载已有 home keyframe")
            await self.send_status(task_id, "EXECUTING", 0.5, message="正在回到 home 姿态")
            await asyncio.to_thread(
                self.run_home,
            )
            await self.send_status(task_id, "SUCCEEDED", 1.0, message="已回到已有 home 姿态")
            return
        if command == "shuffle_scene":
            if not self.direct_pose_execution:
                await self.send_status(
                    task_id,
                    "FAILED",
                    0.0,
                    error_code="SHUFFLE_NOT_SUPPORTED",
                    message="当前仿真配置不支持安全随机布局",
                )
                return
            self.last_detections_by_id.clear()
            await self.send_status(
                task_id,
                "PLANNING",
                0.2,
                message="正在选择下一个已验证安全布局",
            )
            await self.send_status(
                task_id,
                "EXECUTING",
                0.5,
                message="正在回收机械臂并重新打乱四个 YCB 物体",
            )
            result = await asyncio.to_thread(self.randomize_scene)
            await self.send_status(
                task_id,
                "SUCCEEDED",
                1.0,
                message=f"随机布局 {result['layout_index'] + 1} 已完成，正在重新运行 YOLO",
            )
            return
        if command != "pick_place":
            await self.send_status(
                task_id,
                "FAILED",
                0.0,
                error_code="UNSUPPORTED_COMMAND",
                message=f"不支持的仿真命令：{command}",
            )
            return

        object_id = str(payload.get("object_id", "")).strip()
        detection = self.last_detections_by_id.get(object_id)
        if detection is None:
            await self.send_status(
                task_id,
                "FAILED",
                0.0,
                error_code="TARGET_NOT_CURRENTLY_DETECTED",
                message=f"目标 {object_id or '(empty)'} 不在最新可执行检测列表中",
            )
            return
        cube_pos = detection.planner_cube_pos.copy()
        if not (
            SAFE_PICK_X[0] <= float(cube_pos[0]) <= SAFE_PICK_X[1]
            and SAFE_PICK_Y[0] <= float(cube_pos[1]) <= SAFE_PICK_Y[1]
        ):
            await self.send_status(
                task_id,
                "FAILED",
                0.0,
                error_code="OUTSIDE_MULTI_OBJECT_SAFE_ZONE",
                message=(
                    f"目标坐标 ({cube_pos[0]:.3f},{cube_pos[1]:.3f}) "
                    "超出多物体仿真安全区，已拒绝"
                ),
            )
            return
        command_position = payload.get("position_m")
        if isinstance(command_position, (list, tuple)) and len(command_position) == 3:
            delta = float(
                np.linalg.norm(
                    np.asarray(command_position[:2], dtype=float)
                    - detection.stable_world_pos[:2]
                )
            )
            if delta > 0.035:
                await self.send_status(
                    task_id,
                    "FAILED",
                    0.0,
                    error_code="TARGET_MOVED_AFTER_SELECTION",
                    message=f"目标选择后移动了 {delta:.3f}m，请重新选择",
                )
                return
        await self.send_status(
            task_id,
            "PLANNING",
            0.08,
            selected_arm="right",
            message=(
                f"正在按视觉坐标直接抓取 {detection.display_name}"
                if self.direct_pose_execution
                else f"正在把 {detection.display_name} 送入已验证安全取件位"
            ),
        )
        if self.direct_pose_execution:
            cube_pos = detection.planner_cube_pos.copy()
            await asyncio.to_thread(self.prepare_direct_execution)
        else:
            await asyncio.to_thread(self.stage_selected_object, detection.body_name)
            cube_pos = DEFAULT_CUBE_SPAWN.copy()

        # The bridge deliberately serialises execution and assigns the right
        # arm to this compact demo workspace. The left arm is moved to a strict
        # park posture before the pick, avoiding the previous centreline clash.
        selected_arm = "right"
        task = core.PickPlaceTask(
            task_id=task_id,
            object_name=detection.body_name,
            target_name="black_frame",
            cube_pos=cube_pos,
            description=(
                f"upper computer selected {detection.class_name} | "
                f"source={detection.source} | right-arm exclusive zone"
            ),
            requested_arm="right",
        )
        core.write_execution_plan([task])
        await self.send_status(
            task_id,
            "PLANNING",
            0.15,
            selected_arm=selected_arm,
            message=(
                f"已锁定 {detection.display_name}；检测来源={detection.source}；"
                f"{'视觉直接坐标' if self.direct_pose_execution else '安全取件位'}="
                f"({cube_pos[0]:.3f},{cube_pos[1]:.3f},{cube_pos[2]:.3f})"
            ),
        )
        planner = core.BimanualTaskPlanner(
            self.model,
            self.data,
            viewer=self.planner_viewer,
            realtime=self.realtime,
            max_retries=0 if self.direct_pose_execution else 1,
            enable_fallback=False,
            simulate_recovery=False,
            right_adapter_mode="rule",
            strict_inactive_arm_park=True,
            right_rule_config=self.object_rule_configs.get(detection.body_name),
            right_rule_speed_scale=self.rule_speed_scale,
            right_rule_preserve_object_pose=self.direct_pose_execution,
            right_rule_use_task_pose_for_grasp=self.direct_pose_execution,
        )
        await self.send_status(
            task_id,
            "EXECUTING",
            0.45,
            selected_arm=selected_arm,
            message=f"右臂独占执行：抓取 {detection.display_name} 并放入黑色框架",
        )
        self.execution_status_started = True
        self.active_task_started_at = time.monotonic()
        self.last_progress_status_at = self.active_task_started_at
        try:
            results = await asyncio.to_thread(planner.run, [task])
            result = results[0]
        except Exception as exc:
            await self.send_status(
                task_id,
                "FAILED",
                0.0,
                selected_arm=selected_arm,
                error_code="PLANNER_EXCEPTION",
                message=str(exc),
            )
            return
        self.last_execution_metrics = {
            "task_id": task_id,
            "pick_success": bool(result.pick_success),
            "place_success": bool(result.place_success),
            "lift_delta_m": float(result.lift_delta),
            "xy_to_frame_m": float(result.xy_dist),
            "z_margin_m": float(result.z_margin),
            "minimum_interarm_tcp_distance_m": float(
                planner.safety.min_interarm_tcp_dist
            ),
            "park_interarm_tcp_distance_m": float(
                planner.safety.last_park_interarm_tcp_distance_m
            ),
            "final_dangerous_collision_count": int(
                planner.collision.last_snapshot["dangerous_count"]
            ),
        }
        await self.send_status(
            task_id,
            "SUCCEEDED" if result.place_success else "FAILED",
            1.0,
            selected_arm=result.selected_arm,
            error_code=None if result.place_success else "PICK_PLACE_FAILED",
            message=result.message,
        )

    def stage_selected_object(self, selected_body: str) -> None:
        """Visibly swap the selected object into the proven pick slot.

        Both arms remain stationary. The object already occupying the slot is
        lifted and moved to the selected object's former display position, then
        the selected object is lowered into the slot. This avoids pretending
        the fixed-point V18.3 skill is a general arbitrary-pose controller.
        """
        if selected_body not in INITIAL_OBJECT_SPAWNS:
            raise ValueError(f"unsupported simulation object: {selected_body}")
        selected_start = core.get_body_pos(self.model, self.data, selected_body)
        selected_target = DEFAULT_CUBE_SPAWN.copy()
        if selected_body == self.pick_slot_object:
            # After a successful placement the remembered slot object is now
            # inside the black frame. A second request for the same object must
            # visibly reload it instead of assuming it is still at the slot.
            if float(np.linalg.norm(selected_start[:2] - selected_target[:2])) < 0.025:
                return
            lift_height = max(float(selected_start[2]), float(selected_target[2])) + 0.10
            phases = (
                (selected_start, np.array([selected_start[0], selected_start[1], lift_height])),
                (
                    np.array([selected_start[0], selected_start[1], lift_height]),
                    np.array([selected_target[0], selected_target[1], lift_height]),
                ),
                (np.array([selected_target[0], selected_target[1], lift_height]), selected_target),
            )
            for start, target in phases:
                for step in range(1, 31):
                    alpha = step / 30.0
                    alpha = 3.0 * alpha**2 - 2.0 * alpha**3
                    core.set_free_body_pos(
                        self.model,
                        self.data,
                        selected_body,
                        (1.0 - alpha) * start + alpha * target,
                    )
                    self.planner_viewer.sync()
                    if self.realtime and self.viewer is not None:
                        time.sleep(0.01)
            core.sim_steps(
                self.model,
                self.data,
                120,
                viewer=self.planner_viewer,
                realtime=self.realtime,
            )
            return
        occupant = self.pick_slot_object
        occupant_start = core.get_body_pos(self.model, self.data, occupant)
        occupant_target = np.array(
            [selected_start[0], selected_start[1], INITIAL_OBJECT_SPAWNS[occupant][2]],
            dtype=float,
        )
        lift_height = max(float(selected_start[2]), float(occupant_start[2])) + 0.10

        phases = (
            (
                selected_start,
                np.array([selected_start[0], selected_start[1], lift_height]),
                occupant_start,
                np.array([occupant_start[0], occupant_start[1], lift_height]),
            ),
            (
                np.array([selected_start[0], selected_start[1], lift_height]),
                np.array([selected_target[0], selected_target[1], lift_height]),
                np.array([occupant_start[0], occupant_start[1], lift_height]),
                np.array([occupant_target[0], occupant_target[1], lift_height]),
            ),
            (
                np.array([selected_target[0], selected_target[1], lift_height]),
                selected_target,
                np.array([occupant_target[0], occupant_target[1], lift_height]),
                occupant_target,
            ),
        )
        for selected_a, selected_b, occupant_a, occupant_b in phases:
            for step in range(1, 31):
                alpha = step / 30.0
                alpha = 3.0 * alpha**2 - 2.0 * alpha**3
                core.set_free_body_pos(
                    self.model,
                    self.data,
                    selected_body,
                    (1.0 - alpha) * selected_a + alpha * selected_b,
                )
                core.set_free_body_pos(
                    self.model,
                    self.data,
                    occupant,
                    (1.0 - alpha) * occupant_a + alpha * occupant_b,
                )
                mujoco.mj_forward(self.model, self.data)
                self.planner_viewer.sync()
                if self.realtime and self.viewer is not None:
                    time.sleep(0.01)
        core.sim_steps(
            self.model,
            self.data,
            120,
            viewer=self.planner_viewer,
            realtime=self.realtime,
        )
        self.pick_slot_object = selected_body

    def prepare_direct_execution(self) -> None:
        """Optional profile hook before a direct-pose action starts."""
        return

    def randomize_scene(self) -> dict[str, Any]:
        raise RuntimeError("safe scene randomization is not configured")

    def run_home(self) -> None:
        prepare_multi_object_scene(
            self.model,
            self.data,
            viewer=self.planner_viewer,
            realtime=self.realtime,
        )
        self.pick_slot_object = "orange_cube"

    def take_snapshot(self) -> SimulationSnapshot:
        return SimulationSnapshot(
            qpos=self.data.qpos.copy(),
            qvel=self.data.qvel.copy(),
            ctrl=self.data.ctrl.copy(),
            act=self.data.act.copy(),
            mocap_pos=self.data.mocap_pos.copy(),
            mocap_quat=self.data.mocap_quat.copy(),
            time=float(self.data.time),
        )

    def capture_live_state(self) -> None:
        """Called by PlannerViewerProxy in the physics/planner thread."""
        now = time.monotonic()
        if now - self.last_live_capture_at < 1.0 / self.fps:
            return
        self.last_live_capture_at = now
        try:
            snapshot = self.take_snapshot()
            try:
                self.live_states.put_nowait(snapshot)
            except queue.Full:
                try:
                    self.live_states.get_nowait()
                except queue.Empty:
                    pass
                self.live_states.put_nowait(snapshot)
        except Exception as exc:
            print(f"[bridge] live state capture failed: {exc}")

    async def publish_pending_live_frame(self) -> bool:
        latest: SimulationSnapshot | None = None
        while True:
            try:
                latest = self.live_states.get_nowait()
            except queue.Empty:
                break
        if latest is None:
            return False
        self.sync_viewer(latest)
        self.last_message = self.build_message(
            snapshot=latest,
            update_command_target=False,
            run_detection=not self.command_active,
        )
        await self.broadcast(self.last_message)
        return True

    @staticmethod
    def _jpeg_base64(rgb: np.ndarray) -> str:
        stream = BytesIO()
        Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(
            stream,
            format="JPEG",
            quality=72,
            optimize=False,
        )
        return base64.b64encode(stream.getvalue()).decode("ascii")

    def apply_snapshot_to(
        self,
        target_data: mujoco.MjData,
        snapshot: SimulationSnapshot,
    ) -> None:
        target_data.qpos[:] = snapshot.qpos
        target_data.qvel[:] = snapshot.qvel
        target_data.ctrl[:] = snapshot.ctrl
        if target_data.act.size:
            target_data.act[:] = snapshot.act
        if target_data.mocap_pos.size:
            target_data.mocap_pos[:] = snapshot.mocap_pos
            target_data.mocap_quat[:] = snapshot.mocap_quat
        target_data.time = snapshot.time
        mujoco.mj_forward(self.model, target_data)

    def apply_snapshot(self, snapshot: SimulationSnapshot) -> None:
        self.apply_snapshot_to(self.render_data, snapshot)

    def sync_viewer(self, snapshot: SimulationSnapshot | None = None) -> None:
        """Update the passive viewer only from its owning main thread."""
        if self.viewer is None or self.viewer_data is None:
            return
        self.apply_snapshot_to(
            self.viewer_data,
            snapshot if snapshot is not None else self.take_snapshot(),
        )
        self.viewer.sync()

    def build_message(
        self,
        *,
        snapshot: SimulationSnapshot | None = None,
        update_command_target: bool = True,
        run_detection: bool = True,
    ) -> str:
        self.apply_snapshot(snapshot if snapshot is not None else self.take_snapshot())
        rgb = self.detector.render_rgb()
        # The target is locked before execution.  Re-running CPU YOLO on every
        # motion frame costs 1-5 s per frame and starves WebSocket heartbeats.
        # During motion we stream the fresh RGB image without inference, then
        # resume strict YOLO immediately after the terminal state.
        found = self.detector.detect_all(rgb) if run_detection else ()
        found = self.postprocess_detections(found)
        detections: list[dict[str, Any]] = []
        if update_command_target:
            self.last_detections_by_id = {item.object_id: item for item in found}
        for detection in found:
            x0, y0, x1, y1 = detection.bbox
            x, y, z = (float(value) for value in detection.stable_world_pos)
            detections.append(
                {
                    "id": detection.object_id,
                    "class_name": detection.class_name,
                    "confidence": detection.confidence,
                    "bbox_xyxy": [x0, y0, x1, y1],
                    "position_m": [x, y, z],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "source": detection.source,
                    "image_angle_rad": detection.image_angle_rad,
                }
            )
        payload = {
            "type": "detection_batch",
            "payload": {
                "schema_version": "1.0",
                "frame_id": "base_link",
                "stamp_ns": time.time_ns(),
                "image_size": [self.detector.width, self.detector.height],
                "image_b64": self._jpeg_base64(rgb),
                "detections": detections,
                "simulation": {
                    "engine": "MuJoCo",
                    "camera": CAMERA_NAME,
                    "frame_index": self.frame_index,
                    "uses_body_ground_truth": False,
                    "detector": self.detector.status,
                    "execution_strategy": (
                        "strict_yolo_pose_direct_grasp + right_arm_exclusive_workspace"
                        if self.direct_pose_execution
                        else "selected_object_pose + right_arm_exclusive_workspace"
                    ),
                    "available_objects": sorted(self.last_detections_by_id),
                    "live_during_motion": True,
                    "frame_perception": (
                        "strict_yolo"
                        if run_detection
                        else "target_locked_live_video_no_repeat_inference"
                    ),
                },
            },
        }
        self.frame_index += 1
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    async def publish_once(self) -> None:
        self.last_message = self.build_message()
        await self.broadcast(self.last_message)

    def postprocess_detections(self, detections: Any) -> tuple[Any, ...]:
        """Profile hook for calibration that remains explicit in the source."""
        return tuple(detections)


async def run_server(args: argparse.Namespace, viewer: Any = None) -> None:
    ensure_vision_xml(SOURCE_XML, VISION_XML, camera_name=CAMERA_NAME)
    patch_bridge_output_paths()
    core.ensure_output_dir()
    model = mujoco.MjModel.from_xml_path(str(VISION_XML))
    data = mujoco.MjData(model)
    stable = prepare_multi_object_scene(
        model,
        data,
        viewer=viewer,
        realtime=True,
    )
    bridge = MujocoUpperBridge(
        model,
        data,
        fps=args.fps,
        viewer=viewer,
        # Keep the virtual camera stream human-viewable even in headless mode.
        # PlannerViewerProxy supplies the timing/capture hook without a window.
        realtime=True,
        detector_backend=args.detector,
        yolo_weights=args.weights,
        yolo_confidence=args.confidence,
    )

    print("=" * 76)
    print("OpenArm MuJoCo -> Upper Computer Vision Bridge")
    print(f"WebSocket: ws://{args.host}:{args.port}")
    print(f"Camera: {CAMERA_NAME} | stream: {bridge.fps:.1f} FPS | 640x480 JPEG")
    print(f"Scene init: reused V18.3 home keyframe; stable objects={stable}")
    print(f"Perception: {bridge.detector.status} (no body-position detection fallback)")
    print("Control: selected-object right-arm execution + strict left-arm park")
    print("=" * 76)

    async with websockets.serve(
        bridge.handler,
        args.host,
        args.port,
        max_size=4_000_000,
        ping_interval=10,
        ping_timeout=20,
    ):
        interval = 1.0 / bridge.fps
        while viewer is None or viewer.is_running():
            started = time.monotonic()
            await bridge.process_one_command()
            published_live = await bridge.publish_pending_live_frame()
            await bridge.publish_progress_heartbeat()
            if not bridge.command_active:
                for _ in range(max(1, int(0.01 / model.opt.timestep))):
                    mujoco.mj_step(model, data)
                bridge.sync_viewer()
                if not published_live:
                    await bridge.publish_once()
            await asyncio.sleep(max(0.0, interval - (time.monotonic() - started)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MuJoCo virtual-camera WebSocket bridge for OpenArm upper computer"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--detector", choices=("auto", "yolo", "color"), default="auto")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="also open the MuJoCo interactive viewer",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.viewer:
        asyncio.run(run_server(args))
        return

    import mujoco.viewer

    ensure_vision_xml(SOURCE_XML, VISION_XML, camera_name=CAMERA_NAME)
    patch_bridge_output_paths()
    core.ensure_output_dir()
    model = mujoco.MjModel.from_xml_path(str(VISION_XML))
    data = mujoco.MjData(model)
    viewer_data = mujoco.MjData(model)
    # run_server creates its own model because the default headless path is
    # deliberately self-contained.  The viewer variant uses this small local
    # loop instead so the passive viewer and stream share exactly one state.
    bridge = MujocoUpperBridge(
        model,
        data,
        fps=args.fps,
        viewer=None,
        viewer_data=viewer_data,
        realtime=True,
        detector_backend=args.detector,
        yolo_weights=args.weights,
        yolo_confidence=args.confidence,
    )

    async def viewer_server(viewer: Any) -> None:
        bridge.set_viewer(viewer, viewer_data)
        stable = prepare_multi_object_scene(
            model,
            data,
            viewer=bridge.planner_viewer,
            realtime=False,
        )
        bridge.sync_viewer()
        print(f"[bridge] reused V18.3 home scene; stable objects={stable}")
        print(f"[bridge] detector={bridge.detector.status}")
        print(f"[bridge] WebSocket ws://{args.host}:{args.port} (viewer enabled)")
        async with websockets.serve(
            bridge.handler,
            args.host,
            args.port,
            max_size=4_000_000,
            ping_interval=10,
            ping_timeout=20,
        ):
            interval = 1.0 / bridge.fps
            while viewer.is_running():
                started = time.monotonic()
                await bridge.process_one_command()
                published_live = await bridge.publish_pending_live_frame()
                await bridge.publish_progress_heartbeat()
                if not bridge.command_active:
                    for _ in range(max(1, int(0.01 / model.opt.timestep))):
                        mujoco.mj_step(model, data)
                    bridge.sync_viewer()
                    if not published_live:
                        await bridge.publish_once()
                await asyncio.sleep(max(0.0, interval - (time.monotonic() - started)))

    with mujoco.viewer.launch_passive(model, viewer_data) as viewer:
        asyncio.run(viewer_server(viewer))


if __name__ == "__main__":
    main()

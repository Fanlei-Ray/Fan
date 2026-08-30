from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
import argparse
import csv
import json
import sys
import time
from typing import Any, Dict, List, Optional

import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# OpenArm bimanual task planner modular demo V11
#
# 目标：
#   一个脚本完成“两只手各抓一次”的上层任务规划演示。
#
# 展示内容：
#   1. 任务层：两个 PickPlaceTask。
#   2. 调度层：根据 cube_y 自动选择 left/right arm。
#   3. 避让层：active arm 执行前，inactive arm park / hold-safe。
#   4. 动作层：通过统一接口调用 left/right pick_and_place。
#   5. 恢复层：失败后自动 retry；仍失败时可 replan 到另一只手臂兜底。
#   6. 碰撞检测层：读取 MuJoCo runtime contacts，记录 robot-robot / robot-env 风险。
#
# 推荐运行：
#   python scripts\task_planner\run_demo.py
#
# 快速无 viewer 测试：
#   python scripts\task_planner\run_demo.py --no-viewer
#
# 备用：只展示右臂稳定分支：
#   python scripts\task_planner\run_demo.py --right-only
# ============================================================


TASK_PLANNER_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = TASK_PLANNER_DIR.parent
ROOT = SCRIPTS_DIR.parent

# right_rule_pick_place.py and right_pick_place.py live in scripts/.
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    import right_rule_pick_place as right_rule
except Exception as exc:
    raise RuntimeError(
        "导入 scripts/right_rule_pick_place.py 失败。请确认本文件放在项目 scripts 目录下。"
    ) from exc

try:
    # 你项目里这个名字虽然叫 right_pick_place.py，但历史上实际是左臂 rule/IK 脚本。
    import right_pick_place as left_rule
except Exception:
    left_rule = None


XML_PATH = ROOT / "v2" / "demo.xml"
OUTPUT_DIR = ROOT / "outputs" / "bimanual_task_planner_demo"
LOG_PATH = OUTPUT_DIR / "task_log_v11.csv"
PLAN_PATH = OUTPUT_DIR / "execution_plan_v11.csv"
SUMMARY_PATH = OUTPUT_DIR / "task_summary_v11.json"
REPORT_PATH = OUTPUT_DIR / "presentation_report_v11.md"
STATE_MACHINE_PATH = OUTPUT_DIR / "state_machine_v11.mmd"
RUNBOOK_PATH = OUTPUT_DIR / "demo_runbook_v11.txt"
PATH_PLAN_PATH = OUTPUT_DIR / "path_plan_v11.csv"
COLLISION_LOG_PATH = OUTPUT_DIR / "collision_log_v11.csv"


# ============================================================
# Demo task configuration
# ============================================================

SELECT_RIGHT_IF_CUBE_Y_GE = 0.020

# 左臂演示点选 y=0.000，是之前左臂工作空间扫描中比较稳的区域之一。
# 右臂演示点用已经验证成功的 fixed point。
DEFAULT_TASKS = [
    {
        "task_id": "task_left_001",
        "object_name": "orange_cube",
        "target_name": "black_frame",
        "cube_pos": np.array([0.516, 0.000, 1.050], dtype=float),
        "description": "左侧/中间区域任务，planner 应选择左臂",
    },
    {
        "task_id": "task_right_001",
        "object_name": "orange_cube",
        "target_name": "black_frame",
        "cube_pos": np.array([0.516, 0.050, 1.050], dtype=float),
        "description": "右侧区域任务，planner 应选择右臂",
    },
]

RIGHT_ONLY_TASKS = [
    {
        "task_id": "task_right_001",
        "object_name": "orange_cube",
        "target_name": "black_frame",
        "cube_pos": np.array([0.516, 0.050, 1.050], dtype=float),
        "description": "右臂稳定展示任务",
    },
]


# V4 队列模式：同一个 planner 连续调度 4 个任务，用于展示更像“任务队列”的上层系统。
# 为保证现场稳定，左臂任务仍使用 y=0.000，右臂任务使用已验证的 y=0.050。
QUEUE_TASKS = [
    {
        "task_id": "queue_left_001",
        "object_name": "orange_cube",
        "target_name": "black_frame",
        "cube_pos": np.array([0.516, 0.000, 1.050], dtype=float),
        "description": "队列任务 1：左臂 pick-and-place",
    },
    {
        "task_id": "queue_right_001",
        "object_name": "orange_cube",
        "target_name": "black_frame",
        "cube_pos": np.array([0.516, 0.050, 1.050], dtype=float),
        "description": "队列任务 2：右臂 pick-and-place",
    },
    {
        "task_id": "queue_left_002",
        "object_name": "orange_cube",
        "target_name": "black_frame",
        "cube_pos": np.array([0.516, 0.000, 1.050], dtype=float),
        "description": "队列任务 3：左臂再次执行",
    },
    {
        "task_id": "queue_right_002",
        "object_name": "orange_cube",
        "target_name": "black_frame",
        "cube_pos": np.array([0.516, 0.050, 1.050], dtype=float),
        "description": "队列任务 4：右臂再次执行",
    },
]


# Right arm best config from fixed pick/place success.
RIGHT_BEST_CONFIG = {
    "name": "right_best_tcp_x-0.006_y-0.020_z+0.055__j7_m060",
    "site_type": "tcp",
    "pregrasp_offset": np.array([-0.006, -0.020, 0.140], dtype=float),
    "grasp_offset": np.array([-0.006, -0.020, 0.055], dtype=float),
    "preplace_offset": np.array([0.0, 0.0, 0.140], dtype=float),
    "place_offset": np.array([0.0, 0.0, 0.080], dtype=float),
    "joint_biases": {
        "right_joint7_ctrl": -0.060,
    },
}


# Left arm default offsets: use the left_rule script values when available.
# 不强行改 XML/TCP，避免影响之前训练好的左臂 BC。
LEFT_DEFAULT_PREGRASP = np.array([-0.005, 0.000, 0.100], dtype=float)
LEFT_DEFAULT_GRASP = np.array([-0.010, 0.000, -0.005], dtype=float)
LEFT_DEFAULT_PREPLACE = np.array([0.000, 0.000, 0.140], dtype=float)
LEFT_DEFAULT_PLACE = np.array([0.000, 0.000, 0.080], dtype=float)


# ============================================================
# State machine definitions
# ============================================================

class PlannerState(Enum):
    INIT = auto()
    PERCEIVE = auto()
    SELECT_ARM = auto()
    SAFETY_CHECK = auto()
    PLAN_PATH = auto()
    PARK_INACTIVE_ARM = auto()
    EXECUTE_PICK_PLACE = auto()
    RETRY_PICK_PLACE = auto()
    REPLAN_ARM = auto()
    VERIFY = auto()
    DONE = auto()
    FAILED = auto()


class ArmStatus(Enum):
    IDLE = auto()
    ACTIVE = auto()
    PARKED = auto()
    ERROR = auto()


@dataclass
class PickPlaceTask:
    task_id: str
    object_name: str
    target_name: str
    cube_pos: np.ndarray
    description: str = ""
    requested_arm: str = "auto"


@dataclass
class TaskResult:
    task_id: str
    selected_arm: str
    pick_success: bool
    place_success: bool
    lift_delta: float
    final_lift_delta: float
    xy_dist: float
    z_margin: float
    cube_final: np.ndarray
    frame_final: np.ndarray
    message: str = ""


@dataclass
class SafetySnapshot:
    active_arm: str
    inactive_arm: str
    left_tcp: np.ndarray
    right_tcp: np.ndarray
    interarm_tcp_dist: float
    workspace_ok: bool
    message: str = ""


@dataclass
class Waypoint:
    """Planner-level waypoint used for safety-aware path decomposition.

    这不是重型 RRT/OMPL 轨迹规划，而是适合当前 demo 的 waypoint-based path planning。
    底层仍然由左右臂 adapter 执行 IK/rule 动作；planner 层先生成可解释的安全路径。
    """
    waypoint_id: int
    task_id: str
    arm: str
    name: str
    target_pos: np.ndarray
    motion_type: str
    safety_role: str
    expected_state: str
    safe: bool = True
    message: str = ""


# ============================================================
# MuJoCo utilities
# ============================================================

def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def maybe_id(model, obj_type, name: str) -> int:
    return mujoco.mj_name2id(model, obj_type, name)


def get_id(model, obj_type, name: str) -> int:
    obj_id = maybe_id(model, obj_type, name)
    if obj_id == -1:
        raise ValueError(f"找不到对象：{name}")
    return obj_id


def actuator_id(model, name: str) -> int:
    return get_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def body_id(model, name: str) -> int:
    return get_id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def site_id(model, name: str) -> int:
    return get_id(model, mujoco.mjtObj.mjOBJ_SITE, name)


def get_body_pos(model, data, name: str) -> np.ndarray:
    return data.xpos[body_id(model, name)].copy()


def get_site_pos(model, data, name: str) -> np.ndarray:
    return data.site_xpos[site_id(model, name)].copy()


def set_ctrl(model, data, name: str, value: float) -> None:
    aid = actuator_id(model, name)
    low, high = model.actuator_ctrlrange[aid]
    data.ctrl[aid] = np.clip(float(value), low, high)


def sync_position_actuators_to_qpos(model, data) -> None:
    for aid in range(model.nu):
        jid = model.actuator_trnid[aid, 0]
        if jid < 0:
            continue
        qaddr = model.jnt_qposadr[jid]
        low, high = model.actuator_ctrlrange[aid]
        data.ctrl[aid] = np.clip(float(data.qpos[qaddr]), low, high)


def load_home(model, data) -> None:
    key_id = maybe_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        print("[SAFETY] 已加载 home keyframe")
    else:
        print("[SAFETY] 未找到 home keyframe，使用当前默认姿态")
    sync_position_actuators_to_qpos(model, data)
    mujoco.mj_forward(model, data)


def set_free_body_pos(model, data, body_name: str, pos: np.ndarray) -> None:
    bid = body_id(model, body_name)
    if model.body_jntnum[bid] < 1:
        raise RuntimeError(f"{body_name} 没有 freejoint，不能直接设置位置。")

    jid = model.body_jntadr[bid]
    qadr = model.jnt_qposadr[jid]
    dadr = model.jnt_dofadr[jid]

    data.qpos[qadr:qadr + 3] = np.asarray(pos, dtype=float)
    data.qpos[qadr + 3:qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[dadr:dadr + 6] = 0.0
    mujoco.mj_forward(model, data)


def sim_steps(model, data, steps: int, viewer=None, realtime: bool = False) -> bool:
    for _ in range(int(steps)):
        if viewer is not None and not viewer.is_running():
            return False
        start = time.time()
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        if realtime and viewer is not None and not getattr(viewer, "handles_realtime_pacing", False):
            sleep_time = model.opt.timestep - (time.time() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)
    return True


def move_to_ctrl(model, data, targets: Dict[str, float], duration: float = 1.0, viewer=None, realtime: bool = False) -> bool:
    start_ctrl = data.ctrl.copy()
    goal_ctrl = data.ctrl.copy()

    for name, value in targets.items():
        aid = actuator_id(model, name)
        low, high = model.actuator_ctrlrange[aid]
        goal_ctrl[aid] = np.clip(float(value), low, high)

    steps = max(1, int(duration / model.opt.timestep))
    for i in range(steps):
        if viewer is not None and not viewer.is_running():
            return False
        alpha = (i + 1) / steps
        alpha = 3.0 * alpha ** 2 - 2.0 * alpha ** 3
        data.ctrl[:] = (1.0 - alpha) * start_ctrl + alpha * goal_ctrl
        start = time.time()
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        if realtime and viewer is not None and not getattr(viewer, "handles_realtime_pacing", False):
            sleep_time = model.opt.timestep - (time.time() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)
    return True


# ============================================================
# Logging
# ============================================================

class TaskLogger:
    def __init__(self, path: Path):
        self.path = path
        ensure_output_dir()
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "task_id",
                "state",
                "selected_arm",
                "inactive_arm",
                "cube_x",
                "cube_y",
                "cube_z",
                "pick_success",
                "place_success",
                "lift_delta",
                "xy_dist",
                "z_margin",
                "left_tcp_x",
                "left_tcp_y",
                "left_tcp_z",
                "right_tcp_x",
                "right_tcp_y",
                "right_tcp_z",
                "interarm_tcp_dist",
                "workspace_ok",
                "message",
            ])

    def log(
        self,
        task: Optional[PickPlaceTask],
        state: PlannerState,
        selected_arm: str = "",
        result: Optional[TaskResult] = None,
        safety: Optional[SafetySnapshot] = None,
        message: str = "",
    ) -> None:
        cube = np.array([np.nan, np.nan, np.nan], dtype=float) if task is None else task.cube_pos
        left_tcp = np.array([np.nan, np.nan, np.nan], dtype=float) if safety is None else safety.left_tcp
        right_tcp = np.array([np.nan, np.nan, np.nan], dtype=float) if safety is None else safety.right_tcp
        inactive_arm = "" if safety is None else safety.inactive_arm
        interarm_tcp_dist = "" if safety is None else float(safety.interarm_tcp_dist)
        workspace_ok = "" if safety is None else bool(safety.workspace_ok)
        merged_message = message
        if safety is not None and safety.message:
            merged_message = (merged_message + " | " if merged_message else "") + safety.message

        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                "" if task is None else task.task_id,
                state.name,
                selected_arm,
                inactive_arm,
                float(cube[0]),
                float(cube[1]),
                float(cube[2]),
                "" if result is None else bool(result.pick_success),
                "" if result is None else bool(result.place_success),
                "" if result is None else float(result.lift_delta),
                "" if result is None else float(result.xy_dist),
                "" if result is None else float(result.z_margin),
                float(left_tcp[0]),
                float(left_tcp[1]),
                float(left_tcp[2]),
                float(right_tcp[0]),
                float(right_tcp[1]),
                float(right_tcp[2]),
                interarm_tcp_dist,
                workspace_ok,
                merged_message,
            ])

# ============================================================
# Arm adapters
# ============================================================

class SafetyManager:
    """Bimanual coordination layer.

    V4 仍然采用稳妥的顺序协作，不做高风险的双臂同时轨迹规划。
    但它显式维护三件事：
      1. active/inactive arm 状态；
      2. workspace permission，确认任务分配是否合理；
      3. TCP 距离监控，给日志留下“避让/安全检查”的量化证据。
    """

    def __init__(
        self,
        model,
        data,
        viewer=None,
        realtime: bool = False,
        *,
        strict_inactive_arm_park: bool = False,
        minimum_interarm_tcp_distance_m: float = 0.24,
    ):
        self.model = model
        self.data = data
        self.viewer = viewer
        self.realtime = realtime
        self.strict_inactive_arm_park = bool(strict_inactive_arm_park)
        self.minimum_interarm_tcp_distance_m = float(minimum_interarm_tcp_distance_m)
        self.arm_status = {
            "left": ArmStatus.IDLE,
            "right": ArmStatus.IDLE,
        }
        self.min_interarm_tcp_dist = float("inf")
        self.last_park_interarm_tcp_distance_m = float("nan")

    def _site_pos_or_nan(self, names: List[str]) -> np.ndarray:
        for name in names:
            if maybe_id(self.model, mujoco.mjtObj.mjOBJ_SITE, name) != -1:
                return get_site_pos(self.model, self.data, name)
        return np.array([np.nan, np.nan, np.nan], dtype=float)

    def left_tcp(self) -> np.ndarray:
        return self._site_pos_or_nan(["left_gripper_tcp", "left_ee_control_point", "left_tcp"])

    def right_tcp(self) -> np.ndarray:
        return self._site_pos_or_nan(["right_gripper_tcp", "right_ee_control_point", "right_tcp"])

    def compute_interarm_tcp_dist(self) -> float:
        left = self.left_tcp()
        right = self.right_tcp()
        if np.any(np.isnan(left)) or np.any(np.isnan(right)):
            return float("nan")
        dist = float(np.linalg.norm(left - right))
        self.min_interarm_tcp_dist = min(self.min_interarm_tcp_dist, dist)
        return dist

    def workspace_permission(self, active_arm: str, cube_pos: np.ndarray) -> bool:
        predicted = "right" if float(cube_pos[1]) >= SELECT_RIGHT_IF_CUBE_Y_GE else "left"
        return predicted == active_arm

    def snapshot(self, active_arm: str, cube_pos: np.ndarray, message: str = "") -> SafetySnapshot:
        inactive = "right" if active_arm == "left" else "left"
        left = self.left_tcp()
        right = self.right_tcp()
        if np.any(np.isnan(left)) or np.any(np.isnan(right)):
            dist = float("nan")
        else:
            dist = float(np.linalg.norm(left - right))
            self.min_interarm_tcp_dist = min(self.min_interarm_tcp_dist, dist)
        return SafetySnapshot(
            active_arm=active_arm,
            inactive_arm=inactive,
            left_tcp=left,
            right_tcp=right,
            interarm_tcp_dist=dist,
            workspace_ok=self.workspace_permission(active_arm, cube_pos),
            message=message,
        )

    def safety_check(self, active_arm: str, cube_pos: np.ndarray) -> SafetySnapshot:
        snap = self.snapshot(active_arm, cube_pos, message="workspace permission checked")
        print(
            f"[SAFETY_CHECK] active={active_arm}, inactive={snap.inactive_arm}, "
            f"workspace_ok={snap.workspace_ok}, interarm_tcp_dist={snap.interarm_tcp_dist}"
        )
        if not snap.workspace_ok:
            print("[SAFETY_CHECK] warning: selected arm does not match workspace rule")
        return snap

    def park_inactive_arm(self, active_arm: str, cube_pos: np.ndarray) -> SafetySnapshot:
        inactive = "right" if active_arm == "left" else "left"
        self.arm_status[active_arm] = ArmStatus.ACTIVE
        self.arm_status[inactive] = ArmStatus.PARKED

        print(f"[SAFETY] active_arm={active_arm}, inactive_arm={inactive} -> park/hold-safe")

        # Strict simulation mode moves the inactive arm out and up before the
        # active arm crosses the centreline. Defaults stay unchanged for the
        # user's existing V18.3 regression scripts.
        if self.strict_inactive_arm_park:
            if inactive == "left":
                park_targets = {
                    "left_joint1_ctrl": -1.00,
                    "left_joint2_ctrl": -0.50,
                    "left_joint3_ctrl": 0.00,
                    "left_joint4_ctrl": 1.20,
                    "left_finger1_ctrl": get_left_attr("LEFT_FINGER_PRE_OPEN", 0.445),
                }
            else:
                park_targets = {
                    "right_joint1_ctrl": 1.00,
                    "right_joint2_ctrl": 0.50,
                    "right_joint3_ctrl": 0.00,
                    "right_joint4_ctrl": 1.20,
                    "right_finger1_ctrl": right_rule.RIGHT_FINGER_PRE_OPEN,
                }
            # The YCB profile starts with the inactive arm already parked for
            # an unobstructed camera view. Avoid replaying a 1.8 s no-op move.
            already_commanded = all(
                abs(
                    float(self.data.ctrl[actuator_id(self.model, name)])
                    - float(value)
                )
                < 1e-4
                for name, value in park_targets.items()
            )
            if not already_commanded:
                move_to_ctrl(
                    self.model,
                    self.data,
                    park_targets,
                    duration=1.8,
                    viewer=self.viewer,
                    realtime=self.realtime,
                )

        # Baseline mode only opens the inactive gripper and holds home.
        if inactive == "left":
            if maybe_id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_finger1_ctrl") != -1:
                set_ctrl(self.model, self.data, "left_finger1_ctrl", get_left_attr("LEFT_FINGER_PRE_OPEN", 0.445))
        else:
            if maybe_id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_finger1_ctrl") != -1:
                set_ctrl(self.model, self.data, "right_finger1_ctrl", right_rule.RIGHT_FINGER_PRE_OPEN)

        mujoco.mj_forward(self.model, self.data)
        sim_steps(self.model, self.data, steps=120, viewer=self.viewer, realtime=self.realtime)
        snap = self.snapshot(active_arm, cube_pos, message=f"{inactive} arm parked / hold-safe")
        self.last_park_interarm_tcp_distance_m = float(snap.interarm_tcp_dist)
        if (
            self.strict_inactive_arm_park
            and np.isfinite(snap.interarm_tcp_dist)
            and snap.interarm_tcp_dist < self.minimum_interarm_tcp_distance_m
        ):
            self.arm_status[inactive] = ArmStatus.ERROR
            raise RuntimeError(
                "INTERARM_CLEARANCE_TOO_SMALL: "
                f"{snap.interarm_tcp_dist:.3f}m < "
                f"{self.minimum_interarm_tcp_distance_m:.3f}m"
            )
        print(
            f"[SAFETY] status: left={self.arm_status['left'].name}, right={self.arm_status['right'].name}, "
            f"interarm_tcp_dist={snap.interarm_tcp_dist}"
        )
        return snap

    def release_after_task(self) -> None:
        self.arm_status["left"] = ArmStatus.IDLE
        self.arm_status["right"] = ArmStatus.IDLE


class CollisionMonitor:
    """MuJoCo runtime collision/contact monitor.

    说明：
      - 这不是重型全局避障规划器；它是当前 demo 里最实用的一层碰撞检测。
      - 它读取 data.contact，把每次 SAFETY_CHECK / PARK / ACTION 后的接触写入 CSV。
      - gripper-cube、cube-frame 这类任务相关接触默认不算危险。
      - V11 修正了 V8 的误报问题：未识别的 ROBOT_OTHER 只记录为 unknown，不再直接算 dangerous。
      - V11 过滤已知 cell_table_col 与 link4/link5 的模型静态接触，并忽略 2mm 以下数值级轻微穿透。
    """

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.last_snapshot = {
            "n_contacts": 0,
            "dangerous_count": 0,
            "min_contact_dist": float("nan"),
            "contacts": [],
            "category_counts": {},
        }
        ensure_output_dir()
        with open(COLLISION_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "task_id",
                "state",
                "selected_arm",
                "contact_id",
                "geom1",
                "geom2",
                "body1",
                "body2",
                "dist",
                "category",
                "dangerous",
                "message",
            ])

    def _name(self, obj_type, obj_id: int) -> str:
        if obj_id < 0:
            return ""
        name = mujoco.mj_id2name(self.model, obj_type, int(obj_id))
        return "" if name is None else str(name)

    def _geom_name(self, geom_id: int) -> str:
        return self._name(mujoco.mjtObj.mjOBJ_GEOM, geom_id)

    def _body_name_from_geom(self, geom_id: int) -> str:
        if geom_id < 0:
            return ""
        body_id_ = int(self.model.geom_bodyid[int(geom_id)])
        return self._name(mujoco.mjtObj.mjOBJ_BODY, body_id_)

    def _has_any(self, text: str, tokens: List[str]) -> bool:
        t = text.lower()
        return any(tok in t for tok in tokens)

    def _is_known_static_table_contact(self, geom1: str, geom2: str, body1: str, body2: str) -> bool:
        """Filter known baseline contacts from the shipped MuJoCo model.

        In this model, the cell table collision geom can already overlap with
        OpenArm link4/link5 collision geoms in home/park poses. These contacts
        are stable model-contact artifacts and were present before task execution,
        so they should be logged but not counted as runtime dangerous collisions.
        """
        pair_text = f"{geom1} {geom2} {body1} {body2}".lower()

        has_cell_table = "cell_table_col" in pair_text
        has_openarm_link = "openarm_left_link" in pair_text or "openarm_right_link" in pair_text
        has_link4_or_5_collision = (
            "link4_left_collision" in pair_text
            or "link5_left_collision" in pair_text
            or "link4_right_collision" in pair_text
            or "link5_right_collision" in pair_text
        )

        return bool(has_cell_table and has_openarm_link and has_link4_or_5_collision)

    def classify(self, geom1: str, geom2: str, body1: str, body2: str) -> str:
        text1 = f"{geom1} {body1}".lower()
        text2 = f"{geom2} {body2}".lower()

        left1 = "left" in text1
        left2 = "left" in text2
        right1 = "right" in text1
        right2 = "right" in text2

        robot1 = left1 or right1
        robot2 = left2 or right2

        task_tokens = [
            "orange", "cube", "blue_part", "phone", "mouse", "black", "frame"
        ]
        env_tokens = ["table", "floor", "ground", "world", "plane"]

        task1 = self._has_any(text1, task_tokens)
        task2 = self._has_any(text2, task_tokens)
        env1 = self._has_any(text1, env_tokens)
        env2 = self._has_any(text2, env_tokens)

        if (left1 and right2) or (right1 and left2):
            return "ROBOT_ROBOT"

        if robot1 and robot2:
            # 同一条手臂内部接触一般由模型接触规则处理，这里记录但不直接判危险。
            return "ROBOT_SELF_OR_INTERNAL"

        if self._is_known_static_table_contact(geom1, geom2, body1, body2):
            return "ROBOT_ENV_BASELINE_IGNORED"

        if (robot1 and env2) or (robot2 and env1):
            return "ROBOT_ENV"

        if (robot1 and task2) or (robot2 and task1):
            return "ROBOT_TASK_OBJECT"

        if task1 or task2:
            return "TASK_OBJECT_CONTACT"

        if robot1 or robot2:
            return "ROBOT_OTHER"

        return "OTHER"

    def is_dangerous(self, category: str, dist: float) -> bool:
        # MuJoCo contact dist 一般 <= margin，负值表示穿透。
        # robot-task-object 是抓取/放置需要的接触，不作为危险。
        # V11 做两层过滤：
        #   1. 已知模型静态接触 ROBOT_ENV_BASELINE_IGNORED 不算危险。
        #   2. 极小穿透只记录为 numerical touch，不算危险。
        #
        # 阈值说明：
        #   - dist < -0.002 表示穿透超过 2 mm，才标为 dangerous。
        #   - 之前 ee_base_link_left/right 的 -3e-7 m 属于数值级接触，过滤掉。
        penetration_threshold = -0.002

        if category == "ROBOT_ROBOT":
            return bool(dist < penetration_threshold)

        if category == "ROBOT_ENV":
            return bool(dist < penetration_threshold)

        return False

    def snapshot(self, task_id: str, state: str, selected_arm: str, message: str = "") -> Dict[str, Any]:
        mujoco.mj_forward(self.model, self.data)

        contacts = []
        dangerous_count = 0
        category_counts = {}
        min_contact_dist = float("inf")

        with open(COLLISION_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if self.data.ncon == 0:
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    task_id,
                    state,
                    selected_arm,
                    -1,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "NO_CONTACT",
                    False,
                    message,
                ])
            for i in range(int(self.data.ncon)):
                c = self.data.contact[i]
                g1 = int(c.geom1)
                g2 = int(c.geom2)
                geom1 = self._geom_name(g1)
                geom2 = self._geom_name(g2)
                body1 = self._body_name_from_geom(g1)
                body2 = self._body_name_from_geom(g2)
                dist = float(c.dist)
                min_contact_dist = min(min_contact_dist, dist)
                category = self.classify(geom1, geom2, body1, body2)
                dangerous = self.is_dangerous(category, dist)
                dangerous_count += int(dangerous)
                category_counts[category] = category_counts.get(category, 0) + 1
                contacts.append((geom1, geom2, body1, body2, dist, category, dangerous))

                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    task_id,
                    state,
                    selected_arm,
                    i,
                    geom1,
                    geom2,
                    body1,
                    body2,
                    dist,
                    category,
                    dangerous,
                    message,
                ])

        if min_contact_dist == float("inf"):
            min_contact_dist = float("nan")

        category_text = ", ".join(f"{k}={v}" for k, v in sorted(category_counts.items()))
        if not category_text:
            category_text = "NO_CONTACT=1"

        print(
            f"[COLLISION_CHECK] task={task_id}, state={state}, "
            f"contacts={int(self.data.ncon)}, dangerous={dangerous_count}, "
            f"min_contact_dist={min_contact_dist}, categories=[{category_text}]"
        )
        if dangerous_count > 0:
            print("[COLLISION_CHECK] warning: dangerous contact detected, see collision_log_v11.csv")

        self.last_snapshot = {
            "n_contacts": int(self.data.ncon),
            "dangerous_count": int(dangerous_count),
            "min_contact_dist": min_contact_dist,
            "contacts": contacts,
            "category_counts": category_counts,
        }
        return self.last_snapshot

class RightArmAdapter:
    def __init__(
        self,
        model,
        data,
        viewer=None,
        realtime: bool = False,
        preserve_inactive_park: bool = False,
        rule_config: Optional[Dict[str, Any]] = None,
        speed_scale: float = 1.0,
        preserve_object_pose: bool = False,
        post_release_retreat: bool = False,
        use_task_pose_for_grasp: bool = False,
    ):
        self.model = model
        self.data = data
        self.viewer = viewer
        self.realtime = realtime
        self.preserve_inactive_park = bool(preserve_inactive_park)
        self.rule_config = RIGHT_BEST_CONFIG if rule_config is None else rule_config
        self.speed_scale = float(speed_scale)
        self.preserve_object_pose = bool(preserve_object_pose)
        self.post_release_retreat = bool(post_release_retreat)
        self.use_task_pose_for_grasp = bool(use_task_pose_for_grasp)

    def pick_and_place(self, task: PickPlaceTask) -> TaskResult:
        print(f"[RIGHT RULE ADAPTER] pick_and_place: {task.task_id}")
        print("[RIGHT RULE ADAPTER] cube_pos:", task.cube_pos)
        print("[RIGHT RULE ADAPTER] config:", self.rule_config["name"])

        right_rule.FIXED_CUBE_POS = np.asarray(task.cube_pos, dtype=float).copy()
        site_name = right_rule.choose_right_site(self.model)

        result = right_rule.run_trial(
            model=self.model,
            site_name=site_name,
            config=self.rule_config,
            do_place=True,
            viewer=self.viewer,
            realtime=self.realtime,
            data=self.data,
            object_name=task.object_name,
            target_name=task.target_name,
            reset_home=not self.preserve_inactive_park,
            speed_scale=self.speed_scale,
            post_release_retreat=self.post_release_retreat,
            reset_object=not self.preserve_object_pose,
            perceived_object_pos=(
                np.asarray(task.cube_pos, dtype=float)
                if self.use_task_pose_for_grasp
                else None
            ),
        )

        return TaskResult(
            task_id=task.task_id,
            selected_arm="right",
            pick_success=bool(result["pick_success"]),
            place_success=bool(result["place_success"]),
            lift_delta=float(result["lift_delta"]),
            final_lift_delta=float(result["final_lift_delta"]),
            xy_dist=float(result["xy_dist"]),
            z_margin=float(result["z_margin"]),
            cube_final=np.asarray(result["cube_final"], dtype=float).copy(),
            frame_final=np.asarray(result["frame_final"], dtype=float).copy(),
            message="right_rule_pick_place.run_trial",
        )


class RightRLPolicyAdapter:
    """Deploy a trained right-arm PPO policy inside the planner simulation.

    This adapter is intentionally a bottom-layer action executor. The upper-level
    planner still performs BT/FSM scheduling, arm selection, safety checks, path
    planning, collision logging, retry and fallback. The adapter only tries to
    execute one right-arm pick-and-place skill with the learned policy.
    """

    RIGHT_ACTUATORS = [
        "right_joint1_ctrl",
        "right_joint2_ctrl",
        "right_joint3_ctrl",
        "right_joint4_ctrl",
        "right_joint5_ctrl",
        "right_joint6_ctrl",
        "right_joint7_ctrl",
        "right_finger1_ctrl",
        "lifter_ctrl",
    ]

    RIGHT_JOINT_ACTUATORS = [
        "right_joint1_ctrl",
        "right_joint2_ctrl",
        "right_joint3_ctrl",
        "right_joint4_ctrl",
        "right_joint5_ctrl",
        "right_joint6_ctrl",
        "right_joint7_ctrl",
    ]

    def __init__(
        self,
        model,
        data,
        viewer=None,
        realtime: bool = False,
        model_path: Optional[str | Path] = None,
        vecnormalize_path: Optional[str | Path] = None,
        max_steps: int = 450,
        frame_skip: int = 8,
    ):
        self.model = model
        self.data = data
        self.viewer = viewer
        self.realtime = realtime
        self.max_steps = int(max_steps)
        self.frame_skip = int(frame_skip)
        self.model_path = Path(model_path) if model_path else ROOT / "outputs" / "rl_right_pick_place_v16_reward_fix" / "ppo_right_pick_place_v16_final.zip"
        self.vecnormalize_path = Path(vecnormalize_path) if vecnormalize_path else ROOT / "outputs" / "rl_right_pick_place_v16_reward_fix" / "vecnormalize_v16_final.pkl"
        self.action_scale = np.array([0.035] * 7 + [0.055, 0.018], dtype=np.float32)
        self.right_grasp_offset = np.array([-0.006, -0.020, 0.055], dtype=np.float64)
        self.success_xy_threshold = 0.055
        self.success_z_margin = 0.005
        self.lift_success_delta_z = 0.020
        self._policy = None
        self._vecnormalize = None
        self._dummy_vec_env = None
        self._initial_cube_z = 0.0
        self._ever_lifted = False

    def _load_policy_if_needed(self) -> None:
        if self._policy is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"RL model not found: {self.model_path}")

        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
            # The V16 env is used only as a dummy env for VecNormalize stats.
            rl_dir = SCRIPTS_DIR / "rl"
            if str(rl_dir) not in sys.path:
                sys.path.insert(0, str(rl_dir))
            from openarm_right_pick_place_env_v16 import OpenArmRightPickPlaceEnv, RightPickPlaceEnvConfig
        except Exception as exc:
            raise RuntimeError(
                "无法导入 stable_baselines3 或 V16 RL 环境。请确认 scripts\rl 已包含 "
                "openarm_right_pick_place_env_v16.py，并且当前 conda 环境已安装 stable-baselines3。"
            ) from exc

        print("[RIGHT RL ADAPTER] load PPO model:", self.model_path)
        self._policy = PPO.load(str(self.model_path), device="cpu")

        cfg = RightPickPlaceEnvConfig(randomize_cube=True, max_steps=self.max_steps)
        self._dummy_vec_env = DummyVecEnv([lambda: OpenArmRightPickPlaceEnv(cfg)])
        if self.vecnormalize_path.exists():
            print("[RIGHT RL ADAPTER] load VecNormalize:", self.vecnormalize_path)
            self._vecnormalize = VecNormalize.load(str(self.vecnormalize_path), self._dummy_vec_env)
            self._vecnormalize.training = False
            self._vecnormalize.norm_reward = False
        else:
            print("[RIGHT RL ADAPTER] WARNING: VecNormalize not found, using raw obs:", self.vecnormalize_path)
            self._vecnormalize = None

    def _right_joint_qpos(self) -> np.ndarray:
        vals = []
        for act_name in self.RIGHT_JOINT_ACTUATORS:
            aid = actuator_id(self.model, act_name)
            jid = self.model.actuator_trnid[aid, 0]
            qaddr = self.model.jnt_qposadr[jid]
            vals.append(float(self.data.qpos[qaddr]))
        return np.asarray(vals, dtype=np.float32)

    def _right_joint_ctrl(self) -> np.ndarray:
        vals = []
        for act_name in self.RIGHT_JOINT_ACTUATORS:
            aid = actuator_id(self.model, act_name)
            vals.append(float(self.data.ctrl[aid]))
        return np.asarray(vals, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        tcp = get_site_pos(self.model, self.data, "right_gripper_tcp")
        cube = get_body_pos(self.model, self.data, "orange_cube")
        frame = get_body_pos(self.model, self.data, "black_frame")
        grasp_pos = cube + self.right_grasp_offset
        tcp_to_cube = grasp_pos - tcp
        cube_to_frame = frame - cube
        lift_delta = np.array([cube[2] - self._initial_cube_z], dtype=np.float32)
        finger = np.array([float(self.data.ctrl[actuator_id(self.model, "right_finger1_ctrl")])], dtype=np.float32)
        lifter = np.array([float(self.data.ctrl[actuator_id(self.model, "lifter_ctrl")])], dtype=np.float32)
        gripper_closed_hint = np.array([1.0 if finger[0] < 0.12 else 0.0], dtype=np.float32)
        ever_lifted = np.array([1.0 if self._ever_lifted else 0.0], dtype=np.float32)
        obs = np.concatenate(
            [
                tcp.astype(np.float32),
                cube.astype(np.float32),
                frame.astype(np.float32),
                tcp_to_cube.astype(np.float32),
                cube_to_frame.astype(np.float32),
                self._right_joint_qpos(),
                self._right_joint_ctrl(),
                finger,
                lifter,
                lift_delta,
                gripper_closed_hint,
                ever_lifted,
            ]
        )
        return obs.astype(np.float32)

    def _dangerous_contact_count(self) -> int:
        count = 0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
            g2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
            pair = f"{g1} {g2}".lower()
            if "orange_cube" in pair or "black_frame" in pair:
                continue
            if "cell_table_col" in pair and ("link4" in pair or "link5" in pair):
                continue
            if c.dist > -0.002:
                continue
            if "right" in pair and ("left" in pair or "table" in pair or "ground" in pair or "world" in pair):
                count += 1
        return count

    def _place_success(self) -> bool:
        cube = get_body_pos(self.model, self.data, "orange_cube")
        frame = get_body_pos(self.model, self.data, "black_frame")
        xy_dist = float(np.linalg.norm(cube[:2] - frame[:2]))
        z_margin = float(cube[2] - frame[2])
        return bool(self._ever_lifted and xy_dist < self.success_xy_threshold and z_margin > self.success_z_margin)

    def _reset_scene_for_policy(self, task: PickPlaceTask) -> None:
        load_home(self.model, self.data)
        set_free_body_pos(self.model, self.data, task.object_name, np.asarray(task.cube_pos, dtype=float))
        set_ctrl(self.model, self.data, "right_finger1_ctrl", 0.445)
        set_ctrl(self.model, self.data, "lifter_ctrl", 0.0)
        mujoco.mj_forward(self.model, self.data)
        sim_steps(self.model, self.data, 80, viewer=self.viewer, realtime=self.realtime)
        set_free_body_pos(self.model, self.data, task.object_name, np.asarray(task.cube_pos, dtype=float))
        mujoco.mj_forward(self.model, self.data)
        sim_steps(self.model, self.data, 20, viewer=self.viewer, realtime=self.realtime)
        self._initial_cube_z = float(get_body_pos(self.model, self.data, task.object_name)[2])
        self._ever_lifted = False

    def _apply_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, -1.0, 1.0)
        for idx, act_name in enumerate(self.RIGHT_ACTUATORS):
            aid = actuator_id(self.model, act_name)
            low, high = self.model.actuator_ctrlrange[aid]
            self.data.ctrl[aid] = np.clip(float(self.data.ctrl[aid] + action[idx] * self.action_scale[idx]), low, high)

    def pick_and_place(self, task: PickPlaceTask) -> TaskResult:
        self._load_policy_if_needed()
        print(f"[RIGHT RL ADAPTER] pick_and_place: {task.task_id}")
        print("[RIGHT RL ADAPTER] model:", self.model_path)
        print("[RIGHT RL ADAPTER] cube_pos:", task.cube_pos)
        self._reset_scene_for_policy(task)
        max_cube_z = self._initial_cube_z
        dangerous_total = 0
        step_count = 0

        for step in range(self.max_steps):
            raw_obs = self._get_obs()[None, :]
            if self._vecnormalize is not None:
                obs = self._vecnormalize.normalize_obs(raw_obs.copy())
            else:
                obs = raw_obs
            action, _ = self._policy.predict(obs, deterministic=True)
            self._apply_action(np.asarray(action)[0])
            sim_steps(self.model, self.data, self.frame_skip, viewer=self.viewer, realtime=self.realtime)
            step_count = step + 1

            cube = get_body_pos(self.model, self.data, task.object_name)
            max_cube_z = max(max_cube_z, float(cube[2]))
            if float(cube[2] - self._initial_cube_z) > self.lift_success_delta_z:
                self._ever_lifted = True
            dangerous_total += self._dangerous_contact_count()
            if self._place_success():
                break

        cube_final = get_body_pos(self.model, self.data, task.object_name)
        frame_final = get_body_pos(self.model, self.data, task.target_name)
        xy_dist = float(np.linalg.norm(cube_final[:2] - frame_final[:2]))
        z_margin = float(cube_final[2] - frame_final[2])
        lift_delta = float(max_cube_z - self._initial_cube_z)
        final_lift_delta = float(cube_final[2] - self._initial_cube_z)
        place_success = bool(self._place_success())
        pick_success = bool(self._ever_lifted or lift_delta > self.lift_success_delta_z)
        timeout = bool(step_count >= self.max_steps and not place_success)

        print(
            f"[RIGHT RL ADAPTER] result: place={place_success}, pick={pick_success}, "
            f"steps={step_count}, timeout={timeout}, lift={lift_delta:.4f}, "
            f"xy={xy_dist:.4f}, dangerous={dangerous_total}"
        )

        return TaskResult(
            task_id=task.task_id,
            selected_arm="right",
            pick_success=pick_success,
            place_success=place_success,
            lift_delta=lift_delta,
            final_lift_delta=final_lift_delta,
            xy_dist=xy_dist,
            z_margin=z_margin,
            cube_final=cube_final.copy(),
            frame_final=frame_final.copy(),
            message=f"right_rl_policy_v16 steps={step_count} timeout={timeout} dangerous={dangerous_total}",
        )


class RightRLFallbackAdapter:
    """Try right-arm RL first; if it fails, fall back to the validated rule expert."""

    def __init__(
        self,
        model,
        data,
        viewer=None,
        realtime: bool = False,
        model_path: Optional[str | Path] = None,
        vecnormalize_path: Optional[str | Path] = None,
        max_steps: int = 450,
    ):
        self.rl = RightRLPolicyAdapter(model, data, viewer, realtime, model_path, vecnormalize_path, max_steps=max_steps)
        self.rule = RightArmAdapter(model, data, viewer, realtime)

    def pick_and_place(self, task: PickPlaceTask) -> TaskResult:
        print("[RIGHT RL FALLBACK ADAPTER] try RL policy first")
        rl_result = self.rl.pick_and_place(task)
        if rl_result.place_success:
            rl_result.message = "right_rl_policy_v16 success"
            print("[RIGHT RL FALLBACK ADAPTER] RL succeeded; no rule fallback needed")
            return rl_result

        print("[RIGHT RL FALLBACK ADAPTER] RL failed/timeout; switching to rule expert fallback")
        rule_result = self.rule.pick_and_place(task)
        rule_result.message = "right_rl_failed_then_rule_fallback | rl_result=" + rl_result.message
        return rule_result


# ---------- Left arm utilities / adapter ----------

def get_left_attr(name: str, default: Any) -> Any:
    if left_rule is None:
        return default
    return getattr(left_rule, name, default)


def get_left_offset(names: List[str], default: np.ndarray) -> np.ndarray:
    if left_rule is not None:
        for name in names:
            if hasattr(left_rule, name):
                return np.asarray(getattr(left_rule, name), dtype=float)
    return np.asarray(default, dtype=float)


class LeftArmAdapter:
    def __init__(self, model, data, viewer=None, realtime: bool = False):
        self.model = model
        self.data = data
        self.viewer = viewer
        self.realtime = realtime

        self.site_name = get_left_attr("LEFT_SITE_NAME", "left_gripper_tcp")
        self.finger_open = float(get_left_attr("LEFT_FINGER_OPEN", 0.49))
        self.finger_pre_open = float(get_left_attr("LEFT_FINGER_PRE_OPEN", 0.445))
        self.finger_close = float(get_left_attr("LEFT_FINGER_CLOSE", 0.0))
        self.lifter_up = float(get_left_attr("LIFTER_UP", 0.11))
        self.lift_success_delta_z = float(get_left_attr("LIFT_SUCCESS_DELTA_Z", 0.015))

        self.pregrasp_offset = get_left_offset(["CUBE_PREGRASP_OFFSET", "PICK_PREGRASP_OFFSET"], LEFT_DEFAULT_PREGRASP)
        self.grasp_offset = get_left_offset(["CUBE_GRASP_OFFSET", "PICK_GRASP_OFFSET"], LEFT_DEFAULT_GRASP)
        self.preplace_offset = get_left_offset(["FRAME_PREPLACE_OFFSET", "PLACE_PREPLACE_OFFSET", "PLACE_PRE_OFFSET"], LEFT_DEFAULT_PREPLACE)
        self.place_offset = get_left_offset(["FRAME_PLACE_OFFSET", "PLACE_GRASP_OFFSET", "PLACE_OFFSET"], LEFT_DEFAULT_PLACE)

    def solve_ik(self, target_pos: np.ndarray):
        if left_rule is None or not hasattr(left_rule, "solve_left_arm_ik"):
            raise RuntimeError("找不到 left_rule.solve_left_arm_ik，无法执行左臂任务。")

        try:
            result = left_rule.solve_left_arm_ik(
                self.model,
                self.data,
                target_pos,
                site_name=self.site_name,
            )
        except TypeError:
            result = left_rule.solve_left_arm_ik(self.model, self.data, target_pos)

        if not isinstance(result, tuple):
            raise RuntimeError(f"solve_left_arm_ik 返回值异常：{result}")

        if len(result) == 2:
            success, ctrl_targets = result
            return bool(success), ctrl_targets

        if len(result) >= 4:
            success = result[0]
            ctrl_targets = result[-1]
            return bool(success), ctrl_targets

        raise RuntimeError(f"无法识别 solve_left_arm_ik 返回值：{result}")

    def reset_scene(self, cube_pos: np.ndarray, object_name: str = "orange_cube") -> None:
        load_home(self.model, self.data)
        set_free_body_pos(self.model, self.data, object_name, cube_pos)
        set_ctrl(self.model, self.data, "left_finger1_ctrl", self.finger_pre_open)
        set_ctrl(self.model, self.data, "right_finger1_ctrl", right_rule.RIGHT_FINGER_PRE_OPEN)
        set_ctrl(self.model, self.data, "lifter_ctrl", 0.0)
        mujoco.mj_forward(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()
        sim_steps(self.model, self.data, steps=700, viewer=self.viewer, realtime=self.realtime)

        # 放稳后再重置 cube，保证初始状态稳定。
        set_free_body_pos(self.model, self.data, object_name, cube_pos)
        set_ctrl(self.model, self.data, "left_finger1_ctrl", self.finger_pre_open)
        set_ctrl(self.model, self.data, "lifter_ctrl", 0.0)
        mujoco.mj_forward(self.model, self.data)
        sim_steps(self.model, self.data, steps=180, viewer=self.viewer, realtime=self.realtime)

    def ik_move_to(self, target_pos: np.ndarray, label: str, duration: float) -> float:
        print(f"[LEFT ADAPTER] IK move {label}: target={np.array2string(target_pos, precision=4)}")
        success, ctrl_targets = self.solve_ik(target_pos)
        if not success:
            print(f"[LEFT ADAPTER] warning: IK for {label} not fully converged, still executing best ctrl")
        move_to_ctrl(self.model, self.data, ctrl_targets, duration=duration, viewer=self.viewer, realtime=self.realtime)
        sim_steps(self.model, self.data, steps=180, viewer=self.viewer, realtime=self.realtime)
        actual = get_site_pos(self.model, self.data, self.site_name)
        err = float(np.linalg.norm(np.asarray(target_pos, dtype=float) - actual))
        print(f"[LEFT ADAPTER] {label}: actual={np.array2string(actual, precision=4)}, err={err:.4f}")
        return err

    def close_gripper(self, duration: float = 2.5) -> None:
        aid = actuator_id(self.model, "left_finger1_ctrl")
        low, high = self.model.actuator_ctrlrange[aid]
        start_value = float(np.clip(self.finger_pre_open, low, high))
        end_value = float(np.clip(self.finger_close, low, high))
        steps = max(1, int(duration / self.model.opt.timestep))
        print(f"[LEFT ADAPTER] close gripper: {start_value:.4f} -> {end_value:.4f}")
        for i in range(steps):
            if self.viewer is not None and not self.viewer.is_running():
                return
            alpha = (i + 1) / steps
            alpha = 3.0 * alpha ** 2 - 2.0 * alpha ** 3
            self.data.ctrl[aid] = (1.0 - alpha) * start_value + alpha * end_value
            start = time.time()
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None:
                self.viewer.sync()
            if self.realtime and self.viewer is not None and not getattr(self.viewer, "handles_realtime_pacing", False):
                sleep_time = self.model.opt.timestep - (time.time() - start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def pick_and_place(self, task: PickPlaceTask) -> TaskResult:
        print(f"[LEFT ADAPTER] pick_and_place: {task.task_id}")
        print("[LEFT ADAPTER] cube_pos:", task.cube_pos)
        print("[LEFT ADAPTER] site_name:", self.site_name)
        print("[LEFT ADAPTER] pregrasp_offset:", self.pregrasp_offset)
        print("[LEFT ADAPTER] grasp_offset:", self.grasp_offset)

        self.reset_scene(task.cube_pos, task.object_name)

        cube_initial = get_body_pos(self.model, self.data, task.object_name)
        cube_initial_z = float(cube_initial[2])
        print("[LEFT ADAPTER] stable cube_initial:", cube_initial)

        # Open/pre-open.
        move_to_ctrl(
            self.model,
            self.data,
            {"left_finger1_ctrl": self.finger_pre_open},
            duration=1.0,
            viewer=self.viewer,
            realtime=self.realtime,
        )
        sim_steps(self.model, self.data, steps=180, viewer=self.viewer, realtime=self.realtime)

        # Pregrasp.
        cube_now = get_body_pos(self.model, self.data, task.object_name)
        self.ik_move_to(cube_now + self.pregrasp_offset, "pregrasp", duration=2.0)

        # Grasp.
        cube_now = get_body_pos(self.model, self.data, task.object_name)
        self.ik_move_to(cube_now + self.grasp_offset, "grasp", duration=1.5)

        # Close.
        self.close_gripper(duration=2.5)
        sim_steps(self.model, self.data, steps=350, viewer=self.viewer, realtime=self.realtime)

        # Lift.
        move_to_ctrl(
            self.model,
            self.data,
            {"lifter_ctrl": self.lifter_up, "left_finger1_ctrl": self.finger_close},
            duration=1.5,
            viewer=self.viewer,
            realtime=self.realtime,
        )

        max_cube_z = cube_initial_z
        for _ in range(500):
            if self.viewer is not None and not self.viewer.is_running():
                break
            start = time.time()
            mujoco.mj_step(self.model, self.data)
            cube_now = get_body_pos(self.model, self.data, task.object_name)
            max_cube_z = max(max_cube_z, float(cube_now[2]))
            if self.viewer is not None:
                self.viewer.sync()
            if self.realtime and self.viewer is not None and not getattr(self.viewer, "handles_realtime_pacing", False):
                sleep_time = self.model.opt.timestep - (time.time() - start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        cube_after_lift = get_body_pos(self.model, self.data, task.object_name)
        lift_delta = float(max_cube_z - cube_initial_z)
        final_lift_delta = float(cube_after_lift[2] - cube_initial_z)
        pick_success = bool(lift_delta > self.lift_success_delta_z)
        print(f"[LEFT ADAPTER] lift_delta={lift_delta:.4f}, pick_success={pick_success}")

        if pick_success:
            frame_pos = get_body_pos(self.model, self.data, task.target_name)
            self.ik_move_to(frame_pos + self.preplace_offset, "preplace", duration=2.0)

            frame_pos = get_body_pos(self.model, self.data, task.target_name)
            self.ik_move_to(frame_pos + self.place_offset, "place", duration=1.5)

            move_to_ctrl(
                self.model,
                self.data,
                {"left_finger1_ctrl": self.finger_open},
                duration=1.0,
                viewer=self.viewer,
                realtime=self.realtime,
            )
            sim_steps(self.model, self.data, steps=500, viewer=self.viewer, realtime=self.realtime)

        final_cube = get_body_pos(self.model, self.data, task.object_name)
        frame_pos = get_body_pos(self.model, self.data, task.target_name)
        xy_dist = float(np.linalg.norm(final_cube[:2] - frame_pos[:2]))
        z_margin = float(final_cube[2] - frame_pos[2])
        place_success = bool(pick_success and xy_dist < 0.055 and z_margin > 0.005)

        return TaskResult(
            task_id=task.task_id,
            selected_arm="left",
            pick_success=pick_success,
            place_success=place_success,
            lift_delta=lift_delta,
            final_lift_delta=final_lift_delta,
            xy_dist=xy_dist,
            z_margin=z_margin,
            cube_final=final_cube.copy(),
            frame_final=frame_pos.copy(),
            message="left_rule IK adapter",
        )



# ============================================================
# Path planning layer
# ============================================================

class PathPlanner:
    """Waypoint-based safety-aware path planner.

    V11 的重点是把“直接调用 pick_and_place”升级为：
    先生成分阶段路径，再做安全检查，再交给底层 adapter 执行。

    当前版本采用稳健的 waypoint 分解：approach high -> pregrasp -> grasp -> lift -> transfer -> preplace -> place -> retreat。
    后续可以把这里替换成 RRT、轨迹优化或 MoveIt/OMPL。
    """

    def __init__(self, model, data):
        self.model = model
        self.data = data
        ensure_output_dir()
        with open(PATH_PLAN_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "task_id",
                "waypoint_id",
                "arm",
                "name",
                "target_x",
                "target_y",
                "target_z",
                "motion_type",
                "safety_role",
                "expected_state",
                "safe",
                "message",
            ])

    def _left_offsets(self):
        pregrasp = get_left_offset(["CUBE_PREGRASP_OFFSET", "PICK_PREGRASP_OFFSET"], LEFT_DEFAULT_PREGRASP)
        grasp = get_left_offset(["CUBE_GRASP_OFFSET", "PICK_GRASP_OFFSET"], LEFT_DEFAULT_GRASP)
        preplace = get_left_offset(["FRAME_PREPLACE_OFFSET", "PLACE_PREPLACE_OFFSET", "PLACE_PRE_OFFSET"], LEFT_DEFAULT_PREPLACE)
        place = get_left_offset(["FRAME_PLACE_OFFSET", "PLACE_GRASP_OFFSET", "PLACE_OFFSET"], LEFT_DEFAULT_PLACE)
        return pregrasp, grasp, preplace, place

    def _right_offsets(self):
        return (
            np.asarray(RIGHT_BEST_CONFIG["pregrasp_offset"], dtype=float),
            np.asarray(RIGHT_BEST_CONFIG["grasp_offset"], dtype=float),
            np.asarray(RIGHT_BEST_CONFIG["preplace_offset"], dtype=float),
            np.asarray(RIGHT_BEST_CONFIG["place_offset"], dtype=float),
        )

    def _workspace_safe(self, arm: str, pos: np.ndarray) -> bool:
        # Demo 级工作区检查：目标点不能太低，且左右臂只在自身工作区或共享放置区附近执行。
        if float(pos[2]) < 0.95 or float(pos[2]) > 1.35:
            return False

        y = float(pos[1])
        if arm == "left":
            return y <= 0.18
        if arm == "right":
            return y >= -0.18
        return False

    def plan_pick_place_path(self, task: PickPlaceTask, arm: str, cube_pos: np.ndarray) -> List[Waypoint]:
        frame_pos = get_body_pos(self.model, self.data, task.target_name)
        if arm == "left":
            pregrasp_offset, grasp_offset, preplace_offset, place_offset = self._left_offsets()
        else:
            pregrasp_offset, grasp_offset, preplace_offset, place_offset = self._right_offsets()

        transfer_high = np.array([
            0.5 * (cube_pos[0] + frame_pos[0]),
            0.5 * (cube_pos[1] + frame_pos[1]),
            max(cube_pos[2], frame_pos[2]) + 0.22,
        ], dtype=float)

        raw = [
            ("APPROACH_HIGH", cube_pos + np.array([0.0, 0.0, 0.180]), "cartesian_waypoint", "keep above object", "approach"),
            ("PREGRASP", cube_pos + pregrasp_offset, "cartesian_waypoint", "slow approach", "pregrasp"),
            ("GRASP", cube_pos + grasp_offset, "cartesian_waypoint", "contact zone", "grasp"),
            ("LIFT", cube_pos + grasp_offset + np.array([0.0, 0.0, 0.120]), "vertical_lift", "clear table/object", "lift"),
            ("TRANSFER_HIGH", transfer_high, "transfer", "avoid table and inactive arm", "transfer"),
            ("PREPLACE", frame_pos + preplace_offset, "cartesian_waypoint", "approach target from above", "preplace"),
            ("PLACE", frame_pos + place_offset, "cartesian_waypoint", "release zone", "place"),
            ("RETREAT", frame_pos + place_offset + np.array([0.0, 0.0, 0.120]), "vertical_retreat", "leave target safely", "retreat"),
        ]

        waypoints: List[Waypoint] = []
        for i, (name, pos, motion_type, safety_role, expected_state) in enumerate(raw, start=1):
            pos = np.asarray(pos, dtype=float)
            safe = self._workspace_safe(arm, pos)
            msg = "ok" if safe else "workspace or height limit warning"
            waypoints.append(
                Waypoint(
                    waypoint_id=i,
                    task_id=task.task_id,
                    arm=arm,
                    name=name,
                    target_pos=pos,
                    motion_type=motion_type,
                    safety_role=safety_role,
                    expected_state=expected_state,
                    safe=bool(safe),
                    message=msg,
                )
            )
        return waypoints

    def path_is_safe(self, waypoints: List[Waypoint]) -> bool:
        return all(w.safe for w in waypoints)

    def write_path(self, waypoints: List[Waypoint]) -> None:
        with open(PATH_PLAN_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for w in waypoints:
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    w.task_id,
                    w.waypoint_id,
                    w.arm,
                    w.name,
                    float(w.target_pos[0]),
                    float(w.target_pos[1]),
                    float(w.target_pos[2]),
                    w.motion_type,
                    w.safety_role,
                    w.expected_state,
                    bool(w.safe),
                    w.message,
                ])

    def print_path(self, task: PickPlaceTask, arm: str, waypoints: List[Waypoint]) -> None:
        print(f"[PATH_PLANNER] task={task.task_id}, arm={arm}, waypoint_count={len(waypoints)}")
        for w in waypoints:
            print(
                f"  {w.waypoint_id:02d}. {w.name:14s} "
                f"target=({w.target_pos[0]:.3f}, {w.target_pos[1]:.3f}, {w.target_pos[2]:.3f}) "
                f"type={w.motion_type:18s} safe={w.safe} | {w.safety_role}"
            )


# ============================================================
# Unified action library
# ============================================================

class RobotActionLibrary:
    """统一动作接口。

    Planner 不直接关心 IK、TCP、夹爪和 lifter 的细节，只通过这个库调用标准动作。
    后续如果把 left rule 换成 BC policy、把 right rule 换成 RL policy，planner 层不用改。
    """

    def __init__(self, left_arm: LeftArmAdapter, right_arm: RightArmAdapter):
        self.adapters = {
            "left": left_arm,
            "right": right_arm,
        }

    def pick_and_place(self, arm: str, task: PickPlaceTask) -> TaskResult:
        if arm not in self.adapters:
            raise ValueError(f"unknown arm: {arm}")
        print(f"[ACTION_LIB] call pick_and_place(arm={arm}, object={task.object_name}, target={task.target_name})")
        return self.adapters[arm].pick_and_place(task)


# ============================================================
# Planner
# ============================================================

class BimanualTaskPlanner:
    def __init__(
        self,
        model,
        data,
        viewer=None,
        realtime: bool = False,
        max_retries: int = 1,
        enable_fallback: bool = True,
        simulate_recovery: bool = False,
        right_adapter_mode: str = "rule",
        rl_model_path: Optional[str | Path] = None,
        rl_vecnormalize_path: Optional[str | Path] = None,
        rl_max_steps: int = 450,
        strict_inactive_arm_park: bool = False,
        minimum_interarm_tcp_distance_m: float = 0.24,
        right_rule_config: Optional[Dict[str, Any]] = None,
        right_rule_speed_scale: float = 1.0,
        right_rule_preserve_object_pose: bool = False,
        right_rule_post_release_retreat: bool = False,
        right_rule_use_task_pose_for_grasp: bool = False,
    ):
        self.model = model
        self.data = data
        self.viewer = viewer
        self.realtime = realtime
        self.max_retries = int(max_retries)
        self.enable_fallback = bool(enable_fallback)
        self.simulate_recovery = bool(simulate_recovery)
        self.right_adapter_mode = str(right_adapter_mode)
        self._simulated_recovery_done = False
        self.logger = TaskLogger(LOG_PATH)
        self.safety = SafetyManager(
            model,
            data,
            viewer,
            realtime,
            strict_inactive_arm_park=strict_inactive_arm_park,
            minimum_interarm_tcp_distance_m=minimum_interarm_tcp_distance_m,
        )
        self.path_planner = PathPlanner(model, data)
        self.collision = CollisionMonitor(model, data)
        self.left_arm = LeftArmAdapter(model, data, viewer, realtime)
        if self.right_adapter_mode == "rule":
            self.right_arm = RightArmAdapter(
                model,
                data,
                viewer,
                realtime,
                preserve_inactive_park=strict_inactive_arm_park,
                rule_config=right_rule_config,
                speed_scale=right_rule_speed_scale,
                preserve_object_pose=right_rule_preserve_object_pose,
                post_release_retreat=right_rule_post_release_retreat,
                use_task_pose_for_grasp=right_rule_use_task_pose_for_grasp,
            )
        elif self.right_adapter_mode == "rl":
            self.right_arm = RightRLPolicyAdapter(
                model,
                data,
                viewer,
                realtime,
                model_path=rl_model_path,
                vecnormalize_path=rl_vecnormalize_path,
                max_steps=rl_max_steps,
            )
        elif self.right_adapter_mode == "rl_fallback":
            self.right_arm = RightRLFallbackAdapter(
                model,
                data,
                viewer,
                realtime,
                model_path=rl_model_path,
                vecnormalize_path=rl_vecnormalize_path,
                max_steps=rl_max_steps,
            )
        else:
            raise ValueError(f"unknown right_adapter_mode: {self.right_adapter_mode}")
        print(f"[PLANNER CONFIG] right_adapter_mode={self.right_adapter_mode}")
        self.actions = RobotActionLibrary(self.left_arm, self.right_arm)

    def perceive(self, task: PickPlaceTask) -> Dict[str, np.ndarray]:
        # 展示版感知：任务定义给出 cube_pos，同时写入仿真场景。
        print(f"[PERCEIVE] task={task.task_id}, object={task.object_name}, target={task.target_name}")
        print("[PERCEIVE] planned cube_pos:", task.cube_pos)
        return {"cube_pos": task.cube_pos.copy()}

    def select_arm(self, cube_pos: np.ndarray, requested_arm: str = "auto") -> str:
        requested = str(requested_arm).strip().lower()
        if requested in {"left", "right"}:
            arm = requested
        else:
            arm = "right" if float(cube_pos[1]) >= SELECT_RIGHT_IF_CUBE_Y_GE else "left"
        print(f"[SELECT_ARM] cube_y={cube_pos[1]:.3f}, threshold={SELECT_RIGHT_IF_CUBE_Y_GE:.3f} -> {arm}")
        return arm

    def make_failure_result(self, task: PickPlaceTask, selected_arm: str, message: str) -> TaskResult:
        cube_final = get_body_pos(self.model, self.data, task.object_name)
        frame_final = get_body_pos(self.model, self.data, task.target_name)
        xy_dist = float(np.linalg.norm(cube_final[:2] - frame_final[:2]))
        z_margin = float(cube_final[2] - frame_final[2])
        return TaskResult(
            task_id=task.task_id,
            selected_arm=selected_arm,
            pick_success=False,
            place_success=False,
            lift_delta=0.0,
            final_lift_delta=0.0,
            xy_dist=xy_dist,
            z_margin=z_margin,
            cube_final=cube_final.copy(),
            frame_final=frame_final.copy(),
            message=message,
        )

    def execute_one_task(self, task: PickPlaceTask) -> TaskResult:
        print("")
        print("=" * 100)
        print(f"TASK START: {task.task_id}")
        print("description:", task.description)
        print("=" * 100)

        state = PlannerState.INIT
        selected_arm = ""
        result: Optional[TaskResult] = None

        self.logger.log(task, state, selected_arm, message="task init")

        state = PlannerState.PERCEIVE
        obs = self.perceive(task)
        self.logger.log(task, state, selected_arm, message="object pose acquired")

        state = PlannerState.SELECT_ARM
        selected_arm = self.select_arm(obs["cube_pos"], task.requested_arm)
        self.logger.log(task, state, selected_arm, message="arm selected")

        state = PlannerState.SAFETY_CHECK
        safety_snapshot = self.safety.safety_check(selected_arm, obs["cube_pos"])
        collision_summary = self.collision.snapshot(task.task_id, state.name, selected_arm, message="pre-action collision snapshot")
        self.logger.log(
            task,
            state,
            selected_arm,
            safety=safety_snapshot,
            message=f"safety pre-check | collision_dangerous={collision_summary['dangerous_count']}",
        )

        state = PlannerState.PLAN_PATH
        waypoints = self.path_planner.plan_pick_place_path(task, selected_arm, obs["cube_pos"])
        path_safe = self.path_planner.path_is_safe(waypoints)
        self.path_planner.print_path(task, selected_arm, waypoints)
        self.path_planner.write_path(waypoints)
        collision_summary = self.collision.snapshot(task.task_id, state.name, selected_arm, message="path planning collision snapshot")
        self.logger.log(
            task,
            state,
            selected_arm,
            safety=safety_snapshot,
            message=f"path planned, waypoints={len(waypoints)}, path_safe={path_safe}, collision_dangerous={collision_summary['dangerous_count']}",
        )
        if not path_safe:
            result = self.make_failure_result(task, selected_arm, message="PATH_UNSAFE: waypoint safety check failed")
            state = PlannerState.FAILED
            self.logger.log(task, state, selected_arm, result=result, safety=safety_snapshot, message="path unsafe, task rejected before execution")
            return result

        state = PlannerState.PARK_INACTIVE_ARM
        safety_snapshot = self.safety.park_inactive_arm(selected_arm, obs["cube_pos"])
        collision_summary = self.collision.snapshot(task.task_id, state.name, selected_arm, message="after inactive arm park")
        self.logger.log(
            task,
            state,
            selected_arm,
            safety=safety_snapshot,
            message=f"inactive arm parked / safe | collision_dangerous={collision_summary['dangerous_count']}",
        )

        state = PlannerState.EXECUTE_PICK_PLACE
        result = self.execute_with_recovery(
            task=task,
            selected_arm=selected_arm,
            cube_pos=obs["cube_pos"],
            safety_snapshot=safety_snapshot,
        )
        collision_summary = self.collision.snapshot(task.task_id, state.name, result.selected_arm, message="post-action collision snapshot")
        selected_arm = result.selected_arm
        self.safety.release_after_task()

        state = PlannerState.VERIFY
        self.print_result(result)
        verify_message = "success" if result.place_success else "failed"
        self.logger.log(task, state, selected_arm, result=result, message=verify_message)

        state = PlannerState.DONE if result.place_success else PlannerState.FAILED
        self.logger.log(task, state, selected_arm, result=result, message="task done" if result.place_success else "task failed")

        print("=" * 100)
        print(f"TASK END: {task.task_id} -> {state.name}")
        print("=" * 100)
        return result

    def execute_with_recovery(
        self,
        task: PickPlaceTask,
        selected_arm: str,
        cube_pos: np.ndarray,
        safety_snapshot: SafetySnapshot,
    ) -> TaskResult:
        """Execute one pick-and-place with retry and optional fallback re-planning.

        V4 在 V3 基础上增加任务队列、执行计划导出和可控故障注入，用于验证恢复逻辑：
        - 第一次失败：同一只手自动 retry；
        - retry 仍失败：如果开启 fallback，则切换到另一只手臂再尝试；
        - 成功 case 不会额外多跑，保持 V2 的稳定性。
        """
        state = PlannerState.EXECUTE_PICK_PLACE
        last_result: Optional[TaskResult] = None

        for attempt in range(self.max_retries + 1):
            attempt_id = attempt + 1
            message = f"start pick_and_place attempt={attempt_id}, arm={selected_arm}"
            self.logger.log(task, state, selected_arm, safety=safety_snapshot, message=message)
            print(f"[RECOVERY] {message}")

            # V4 可控故障注入：用于验证 RETRY_PICK_PLACE 日志和恢复逻辑。
            # 这个故障只在第一个任务的第一次尝试触发一次，不移动真实机械臂；随后 planner 会自动重试并执行真实动作。
            if self.simulate_recovery and (not self._simulated_recovery_done) and attempt == 0:
                self._simulated_recovery_done = True
                last_result = self.make_failure_result(
                    task,
                    selected_arm,
                    message="SIMULATED_FAILURE: injected failure before real action",
                )
                safety_snapshot = self.safety.snapshot(
                    selected_arm,
                    cube_pos,
                    message="post simulated failure safety snapshot",
                )
                self.logger.log(
                    task,
                    state,
                    selected_arm,
                    result=last_result,
                    safety=safety_snapshot,
                    message="simulated pick_and_place failure injected",
                )
                retry_state = PlannerState.RETRY_PICK_PLACE
                self.logger.log(
                    task,
                    retry_state,
                    selected_arm,
                    result=last_result,
                    safety=safety_snapshot,
                    message="retry scheduled after simulated failure",
                )
                print("[RECOVERY] simulated failure injected; retry same arm with real action")
                continue

            last_result = self.actions.pick_and_place(selected_arm, task)
            safety_snapshot = self.safety.snapshot(
                selected_arm,
                cube_pos,
                message=f"post action safety snapshot, attempt={attempt_id}",
            )
            self.logger.log(
                task,
                state,
                selected_arm,
                result=last_result,
                safety=safety_snapshot,
                message="pick_and_place finished",
            )

            if last_result.place_success:
                return last_result

            if attempt < self.max_retries:
                retry_state = PlannerState.RETRY_PICK_PLACE
                self.logger.log(
                    task,
                    retry_state,
                    selected_arm,
                    result=last_result,
                    safety=safety_snapshot,
                    message=f"retry scheduled after failed attempt={attempt_id}",
                )
                print(f"[RECOVERY] failed attempt={attempt_id}; retry same arm={selected_arm}")

        assert last_result is not None

        if not self.enable_fallback:
            return last_result

        fallback_arm = "right" if selected_arm == "left" else "left"
        replan_state = PlannerState.REPLAN_ARM
        self.logger.log(
            task,
            replan_state,
            fallback_arm,
            result=last_result,
            safety=safety_snapshot,
            message=f"fallback replan: {selected_arm} -> {fallback_arm}",
        )
        print(f"[RECOVERY] fallback replan: {selected_arm} -> {fallback_arm}")

        # 切换执行手臂前，重新做 safety check 和 inactive arm park。
        safety_snapshot = self.safety.safety_check(fallback_arm, cube_pos)
        self.logger.log(
            task,
            PlannerState.SAFETY_CHECK,
            fallback_arm,
            safety=safety_snapshot,
            message="fallback safety pre-check",
        )
        safety_snapshot = self.safety.park_inactive_arm(fallback_arm, cube_pos)
        self.logger.log(
            task,
            PlannerState.PARK_INACTIVE_ARM,
            fallback_arm,
            safety=safety_snapshot,
            message="fallback inactive arm parked / safe",
        )

        self.logger.log(
            task,
            state,
            fallback_arm,
            safety=safety_snapshot,
            message="start fallback pick_and_place",
        )
        fallback_result = self.actions.pick_and_place(fallback_arm, task)
        safety_snapshot = self.safety.snapshot(
            fallback_arm,
            cube_pos,
            message="post fallback action safety snapshot",
        )
        self.logger.log(
            task,
            state,
            fallback_arm,
            result=fallback_result,
            safety=safety_snapshot,
            message="fallback pick_and_place finished",
        )
        return fallback_result

    @staticmethod
    def print_result(result: TaskResult) -> None:
        print("")
        print("-" * 80)
        print(f"VERIFY RESULT: {result.task_id}")
        print("-" * 80)
        print("selected_arm:", result.selected_arm)
        print("pick_success:", result.pick_success)
        print("place_success:", result.place_success)
        print("lift_delta:", result.lift_delta)
        print("final_lift_delta:", result.final_lift_delta)
        print("xy_dist:", result.xy_dist)
        print("z_margin:", result.z_margin)
        print("cube_final:", result.cube_final)
        print("frame_final:", result.frame_final)
        print("message:", result.message)
        print("-" * 80)

    def run(self, tasks: List[PickPlaceTask]) -> List[TaskResult]:
        print_presentation_header(tasks)
        results: List[TaskResult] = []
        for i, task in enumerate(tasks, start=1):
            print(f"\n[PLANNER] Running task {i}/{len(tasks)}")
            result = self.execute_one_task(task)
            results.append(result)

            if i < len(tasks):
                print("[PLANNER] pause 1.5s before next task")
                sim_steps(self.model, self.data, steps=int(1.5 / self.model.opt.timestep), viewer=self.viewer, realtime=self.realtime)

        print_final_summary(results)
        write_summary_json(results)
        write_presentation_assets(tasks, results)
        return results


# ============================================================
# Presentation helpers
# ============================================================

def make_tasks(raw_tasks: List[Dict[str, Any]]) -> List[PickPlaceTask]:
    return [
        PickPlaceTask(
            task_id=str(t["task_id"]),
            object_name=str(t.get("object_name", "orange_cube")),
            target_name=str(t.get("target_name", "black_frame")),
            cube_pos=np.asarray(t["cube_pos"], dtype=float),
            description=str(t.get("description", "")),
        )
        for t in raw_tasks
    ]


def predicted_arm(cube_pos: np.ndarray) -> str:
    return "right" if float(cube_pos[1]) >= SELECT_RIGHT_IF_CUBE_Y_GE else "left"


def write_execution_plan(tasks: List[PickPlaceTask]) -> None:
    ensure_output_dir()
    with open(PLAN_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "order",
            "task_id",
            "object_name",
            "target_name",
            "cube_x",
            "cube_y",
            "cube_z",
            "planned_arm",
            "inactive_arm",
            "planned_states",
            "description",
        ])
        for i, task in enumerate(tasks, start=1):
            arm = predicted_arm(task.cube_pos)
            inactive = "right" if arm == "left" else "left"
            writer.writerow([
                i,
                task.task_id,
                task.object_name,
                task.target_name,
                float(task.cube_pos[0]),
                float(task.cube_pos[1]),
                float(task.cube_pos[2]),
                arm,
                inactive,
                "INIT>PERCEIVE>SELECT_ARM>SAFETY_CHECK>PLAN_PATH>PARK_INACTIVE_ARM>EXECUTE_PICK_PLACE>VERIFY>DONE",
                task.description,
            ])


def write_summary_json(results: List[TaskResult]) -> None:
    ensure_output_dir()
    payload = {
        "demo": "openarm_bimanual_task_planner_demo_v6_path_planning",
        "total_tasks": len(results),
        "success_count": sum(1 for r in results if r.place_success),
        "selector_rule": f"right if cube_y >= {SELECT_RIGHT_IF_CUBE_Y_GE:.3f}, else left",
        "log_path": str(LOG_PATH),
        "plan_path": str(PLAN_PATH),
        "path_plan_path": str(PATH_PLAN_PATH),
        "collision_log_path": str(COLLISION_LOG_PATH),
        "results": [
            {
                "task_id": r.task_id,
                "selected_arm": r.selected_arm,
                "pick_success": bool(r.pick_success),
                "place_success": bool(r.place_success),
                "lift_delta": float(r.lift_delta),
                "xy_dist": float(r.xy_dist),
                "z_margin": float(r.z_margin),
                "message": r.message,
            }
            for r in results
        ],
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def print_presentation_header(tasks: List[PickPlaceTask]) -> None:
    print("=" * 100)
    print("OpenArm 双臂上层任务规划 Demo V11 Collision Filter")
    print("=" * 100)
    print("核心演示：")
    print("  1. 单一 planner 接收任务序列，不再手动分开运行左右臂脚本。")
    print("  2. planner 根据 cube_y 自动选择 left/right arm。")
    print("  3. active arm 执行时，inactive arm 进入 park/hold-safe 状态，作为基础避让。")
    print("  4. 支持任务队列导出、执行日志、summary JSON。")
    print("  5. 可选 --simulate-recovery，用故障注入验证 retry 恢复逻辑。")
    print("  6. V11 增加 MuJoCo contact 碰撞检测日志 collision_log_v11.csv。")
    print("  7. V6/V11 包含 PathPlanner：生成 waypoint path、做 path safety check、导出 path_plan_v11.csv。")
    print("")
    print("State machine:")
    print("  INIT -> PERCEIVE -> SELECT_ARM -> SAFETY_CHECK -> PLAN_PATH -> COLLISION_CHECK")
    print("       -> PARK_INACTIVE_ARM -> EXECUTE_PICK_PLACE -> [RETRY_PICK_PLACE] -> [REPLAN_ARM] -> VERIFY -> DONE/FAILED")
    print("")
    print("Behavior tree view:")
    print("  Sequence(TaskQueue)")
    print("    ├─ PerceiveObject")
    print("    ├─ SelectArm")
    print("    ├─ SafetyCheck")
    print("    ├─ ParkInactiveArm")
    print("    ├─ PickAndPlace(active_arm)")
    print("    ├─ RetryIfNeeded")
    print("    ├─ ReplanFallbackArmIfNeeded")
    print("    └─ VerifySuccess")
    print("")
    print(f"selector: right if cube_y >= {SELECT_RIGHT_IF_CUBE_Y_GE:.3f}, else left")
    print("XML_PATH:", XML_PATH)
    print("LOG_PATH:", LOG_PATH)
    print("PLAN_PATH:", PLAN_PATH)
    print("PATH_PLAN_PATH:", PATH_PLAN_PATH)
    print("COLLISION_LOG_PATH:", COLLISION_LOG_PATH)
    print("SUMMARY_PATH:", SUMMARY_PATH)
    print("REPORT_PATH:", REPORT_PATH)
    print("STATE_MACHINE_PATH:", STATE_MACHINE_PATH)
    print("RUNBOOK_PATH:", RUNBOOK_PATH)
    print("")
    print("Tasks:")
    for task in tasks:
        arm = predicted_arm(task.cube_pos)
        print(
            f"  {task.task_id:16s} cube=({task.cube_pos[0]:.3f}, {task.cube_pos[1]:.3f}, {task.cube_pos[2]:.3f}) "
            f"-> selected_arm={arm:5s} | {task.description}"
        )
    print("=" * 100)


def print_final_summary(results: List[TaskResult]) -> None:
    print("")
    print("=" * 100)
    print("Bimanual task planner demo 总结")
    print("=" * 100)
    success_count = 0
    for result in results:
        if result.place_success:
            success_count += 1
        print(
            f"{result.task_id:16s} "
            f"arm={result.selected_arm:5s} "
            f"pick={result.pick_success} "
            f"place={result.place_success} "
            f"lift={result.lift_delta:.4f} "
            f"xy={result.xy_dist:.4f} "
            f"z_margin={result.z_margin:.4f}"
        )
    print("")
    print(f"total place_success: {success_count}/{len(results)}")
    print("LOG_PATH:", LOG_PATH)
    print("PLAN_PATH:", PLAN_PATH)
    print("PATH_PLAN_PATH:", PATH_PLAN_PATH)
    print("COLLISION_LOG_PATH:", COLLISION_LOG_PATH)
    print("SUMMARY_PATH:", SUMMARY_PATH)
    print("REPORT_PATH:", REPORT_PATH)
    print("STATE_MACHINE_PATH:", STATE_MACHINE_PATH)
    print("RUNBOOK_PATH:", RUNBOOK_PATH)
    print("=" * 100)
    if success_count == len(results):
        print("结论：双臂任务规划、自动选臂、基础避让、连续执行与失败恢复框架已完成。")
    else:
        print("说明：上层 planner 与右臂分支已可演示；失败分支可作为后续 left TCP/BC 接入优化点。")


def write_presentation_assets(tasks: List[PickPlaceTask], results: List[TaskResult]) -> None:
    """Generate human-readable presentation files after a run."""
    ensure_output_dir()
    total = len(results)
    success_count = sum(1 for r in results if r.place_success)
    success_rate = (success_count / total) if total else 0.0

    # 1) Mermaid state-machine diagram.
    state_machine = """stateDiagram-v2
    [*] --> INIT
    INIT --> PERCEIVE
    PERCEIVE --> SELECT_ARM
    SELECT_ARM --> SAFETY_CHECK
    SAFETY_CHECK --> PLAN_PATH: workspace_ok
    PLAN_PATH --> PARK_INACTIVE_ARM: path_safe
    SAFETY_CHECK --> FAILED: unsafe
    PARK_INACTIVE_ARM --> EXECUTE_PICK_PLACE
    EXECUTE_PICK_PLACE --> VERIFY: success
    EXECUTE_PICK_PLACE --> RETRY_PICK_PLACE: failure and retry left
    RETRY_PICK_PLACE --> EXECUTE_PICK_PLACE
    EXECUTE_PICK_PLACE --> REPLAN_ARM: retry failed and fallback enabled
    REPLAN_ARM --> SAFETY_CHECK
    VERIFY --> DONE: pick_success && place_success
    VERIFY --> FAILED: otherwise
    DONE --> [*]
    FAILED --> [*]
"""
    with open(STATE_MACHINE_PATH, "w", encoding="utf-8") as f:
        f.write(state_machine)

    # 2) Runbook for live demo.
    runbook = f"""OpenArm 双臂任务规划 Demo V11 运行手册

推荐现场命令：
  cd /d {ROOT}
  python scripts\\openarm_bimanual_task_planner_demo_v11_collision_detection.py

恢复逻辑演示命令：
  python scripts\\openarm_bimanual_task_planner_demo_v11_collision_detection.py --simulate-recovery

队列演示命令：
  python scripts\\openarm_bimanual_task_planner_demo_v11_collision_detection.py --queue-demo

无 viewer 快速测试：
  python scripts\\openarm_bimanual_task_planner_demo_v11_collision_detection.py --no-viewer

输出文件：
  task log:      {LOG_PATH}
  execution plan:{PLAN_PATH}
  path plan:     {PATH_PLAN_PATH}
  collision log: {COLLISION_LOG_PATH}
  summary json:  {SUMMARY_PATH}
  report:        {REPORT_PATH}
  state machine: {STATE_MACHINE_PATH}

现场讲解顺序：
  1. 说明这是一个单一上层 planner，不是左右手分别运行脚本。
  2. 说明 planner 根据 cube_y 自动选择 left/right arm。
  3. 指出 active arm 执行任务前，inactive arm 进入 PARK_INACTIVE_ARM 状态。
  4. 展示终端状态流：INIT -> PERCEIVE -> SELECT_ARM -> SAFETY_CHECK -> PARK_INACTIVE_ARM -> EXECUTE_PICK_PLACE -> VERIFY -> DONE。
  5. 展示最终 total place_success。
"""
    with open(RUNBOOK_PATH, "w", encoding="utf-8") as f:
        f.write(runbook)

    # 3) Markdown presentation report.
    rows = []
    for r in results:
        rows.append(
            f"| {r.task_id} | {r.selected_arm} | {r.pick_success} | {r.place_success} | "
            f"{r.lift_delta:.4f} | {r.xy_dist:.4f} | {r.z_margin:.4f} | {r.message} |"
        )
    rows_text = "\n".join(rows)

    plan_rows = []
    for i, t in enumerate(tasks, start=1):
        arm = predicted_arm(t.cube_pos)
        inactive = "right" if arm == "left" else "left"
        plan_rows.append(
            f"| {i} | {t.task_id} | ({t.cube_pos[0]:.3f}, {t.cube_pos[1]:.3f}, {t.cube_pos[2]:.3f}) | "
            f"{arm} | {inactive} | {t.description} |"
        )
    plan_text = "\n".join(plan_rows)

    report = f"""# OpenArm 双臂上层任务规划系统 Demo V11 汇报记录

## 一、Demo 目标

本 demo 展示一个统一的双臂上层任务规划系统。系统接收 pick-and-place 任务队列，自动读取任务中的物体位置，根据工作区规则选择左臂或右臂，并在一只机械臂执行时让另一只机械臂进入 park/hold-safe 状态，实现基础双臂避让和任务调度。

## 二、系统结构

- **任务层**：`PickPlaceTask(object=orange_cube, target=black_frame)`
- **规划层**：状态机 `INIT -> PERCEIVE -> SELECT_ARM -> SAFETY_CHECK -> PARK_INACTIVE_ARM -> EXECUTE_PICK_PLACE -> VERIFY -> DONE`
- **安全层**：`SafetyManager` 记录 inactive arm、TCP 距离、workspace_ok
- **动作层**：`RobotActionLibrary.pick_and_place(arm, object, target)` 统一封装左/右臂调用
- **恢复层**：支持 retry 和 fallback replan
- **日志层**：输出 task log、execution plan、summary JSON 和本 report

## 三、自动选臂规则

```text
if cube_y >= {SELECT_RIGHT_IF_CUBE_Y_GE:.3f}: use right arm
else: use left arm
```

## 四、执行计划

| Order | Task ID | Cube Position | Planned Arm | Inactive Arm | Description |
|---:|---|---|---|---|---|
{plan_text}

## 五、路径规划输出

V11 会为每个任务生成 waypoint-based path plan，保存到：`{PATH_PLAN_PATH}`。每个 waypoint 包含目标位置、motion_type、safety_role、expected_state 和 safe 标记。

## 六、执行结果

| Task ID | Arm | Pick Success | Place Success | Lift Delta | XY Dist | Z Margin | Message |
|---|---|---:|---:|---:|---:|---:|---|
{rows_text}

## 七、结果汇总

- Total tasks: **{total}**
- Place success: **{success_count}/{total}**
- Success rate: **{success_rate:.1%}**

## 八、可展示亮点

1. 单一 planner 连续调度左右臂，不再分别运行左右臂脚本。
2. 系统根据物体位置自动选择机械臂。
3. active arm 执行时 inactive arm 自动进入安全等待状态。
4. 支持状态机日志、行为树视图、执行计划、路径计划和 summary JSON。
5. 支持故障注入与 retry 恢复逻辑，能够展示上层智能调度雏形。

## 九、生成文件

- Task log: `{LOG_PATH}`
- Execution plan: `{PLAN_PATH}`
- Path plan: `{PATH_PLAN_PATH}`
- Collision log: `{COLLISION_LOG_PATH}`
- Summary JSON: `{SUMMARY_PATH}`
- State machine Mermaid: `{STATE_MACHINE_PATH}`
- Runbook: `{RUNBOOK_PATH}`
"""
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print("[V11] 已生成汇报材料：")
    print("[V11] REPORT_PATH:", REPORT_PATH)
    print("[V11] STATE_MACHINE_PATH:", STATE_MACHINE_PATH)
    print("[V11] RUNBOOK_PATH:", RUNBOOK_PATH)


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="OpenArm bimanual task planner modular demo V11 with waypoint path planning")
    parser.add_argument("--no-viewer", action="store_true", help="不打开 viewer，快速 headless 测试")
    parser.add_argument("--right-only", action="store_true", help="只运行右臂稳定展示任务")
    parser.add_argument("--queue-demo", action="store_true", help="运行 4 个任务的队列演示：left/right/left/right")
    parser.add_argument("--dry-run-plan", action="store_true", help="只生成 execution_plan_v11.csv，不执行机械臂动作")
    parser.add_argument("--simulate-recovery", action="store_true", help="在第一个任务第一次尝试时注入一次模拟失败，展示 RETRY_PICK_PLACE 恢复逻辑")
    parser.add_argument("--max-retries", type=int, default=1, help="失败后同一只手最多重试次数，默认 1")
    parser.add_argument("--no-fallback", dest="enable_fallback", action="store_false", help="失败后不切换到另一只手兜底")
    parser.add_argument("--no-hold-viewer", dest="hold_viewer", action="store_false", help="结束后不保持 viewer")
    parser.set_defaults(hold_viewer=True, enable_fallback=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not XML_PATH.exists():
        raise FileNotFoundError(f"找不到 XML：{XML_PATH}")

    ensure_output_dir()

    if args.right_only:
        raw_tasks = RIGHT_ONLY_TASKS
    elif args.queue_demo:
        raw_tasks = QUEUE_TASKS
    else:
        raw_tasks = DEFAULT_TASKS

    tasks = make_tasks(raw_tasks)
    write_execution_plan(tasks)

    if args.dry_run_plan:
        print_presentation_header(tasks)
        print("[DRY_RUN_PLAN] 已生成执行计划，不执行机械臂动作。")
        print("PLAN_PATH:", PLAN_PATH)
        print("PATH_PLAN_PATH:", PATH_PLAN_PATH)
        print("COLLISION_LOG_PATH:", COLLISION_LOG_PATH)
        return

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    if args.no_viewer:
        planner = BimanualTaskPlanner(
            model,
            data,
            viewer=None,
            realtime=False,
            max_retries=args.max_retries,
            enable_fallback=args.enable_fallback,
            simulate_recovery=args.simulate_recovery,
        )
        planner.run(tasks)
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        print("viewer 已打开。2 秒后开始双臂任务规划 demo。")
        time.sleep(2.0)

        planner = BimanualTaskPlanner(
            model,
            data,
            viewer=viewer,
            realtime=True,
            max_retries=args.max_retries,
            enable_fallback=args.enable_fallback,
            simulate_recovery=args.simulate_recovery,
        )
        planner.run(tasks)

        if args.hold_viewer:
            print("viewer 保持运行。关闭 viewer 或 Ctrl+C 结束。")
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()

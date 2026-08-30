from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
import argparse
import csv
import sys
import time
from typing import Any, Dict, List, Optional

import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# OpenArm bimanual task planner demo
#
# 目标：
#   一个脚本完成“两只手各抓一次”的上层任务规划演示。
#
# 展示内容：
#   1. 任务层：两个 PickPlaceTask。
#   2. 调度层：根据 cube_y 自动选择 left/right arm。
#   3. 避让层：active arm 执行前，inactive arm park / hold-safe。
#   4. 动作层：通过统一接口调用 left/right pick_and_place。
#
# 推荐运行：
#   python scripts\openarm_bimanual_task_planner_demo.py
#
# 快速无 viewer 测试：
#   python scripts\openarm_bimanual_task_planner_demo.py --no-viewer
#
# 备用：只展示右臂稳定分支：
#   python scripts\openarm_bimanual_task_planner_demo.py --right-only
# ============================================================


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

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
LOG_PATH = OUTPUT_DIR / "task_log.csv"


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
    PARK_INACTIVE_ARM = auto()
    EXECUTE_PICK_PLACE = auto()
    VERIFY = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class PickPlaceTask:
    task_id: str
    object_name: str
    target_name: str
    cube_pos: np.ndarray
    description: str = ""


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
        if realtime and viewer is not None:
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
        if realtime and viewer is not None:
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
                "cube_x",
                "cube_y",
                "cube_z",
                "pick_success",
                "place_success",
                "lift_delta",
                "xy_dist",
                "z_margin",
                "message",
            ])

    def log(self, task: Optional[PickPlaceTask], state: PlannerState, selected_arm: str = "", result: Optional[TaskResult] = None, message: str = "") -> None:
        cube = np.array([np.nan, np.nan, np.nan], dtype=float) if task is None else task.cube_pos
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                "" if task is None else task.task_id,
                state.name,
                selected_arm,
                float(cube[0]),
                float(cube[1]),
                float(cube[2]),
                "" if result is None else bool(result.pick_success),
                "" if result is None else bool(result.place_success),
                "" if result is None else float(result.lift_delta),
                "" if result is None else float(result.xy_dist),
                "" if result is None else float(result.z_margin),
                message,
            ])


# ============================================================
# Arm adapters
# ============================================================

class SafetyManager:
    """Simple bimanual safety layer: mutual exclusion + inactive arm park/home."""

    def __init__(self, model, data, viewer=None, realtime: bool = False):
        self.model = model
        self.data = data
        self.viewer = viewer
        self.realtime = realtime

    def park_inactive_arm(self, active_arm: str) -> None:
        inactive = "right" if active_arm == "left" else "left"
        print(f"[SAFETY] active_arm={active_arm}, inactive_arm={inactive} -> park/hold-safe")

        # 展示版避让策略：把非工作臂相关夹爪打开，并尽量保持 home / 当前安全位。
        # 不强行移动整条手臂，避免破坏已经验证过的 home/IK 初始化。
        if inactive == "left":
            if maybe_id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_finger1_ctrl") != -1:
                set_ctrl(self.model, self.data, "left_finger1_ctrl", get_left_attr("LEFT_FINGER_PRE_OPEN", 0.445))
        else:
            if maybe_id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_finger1_ctrl") != -1:
                set_ctrl(self.model, self.data, "right_finger1_ctrl", right_rule.RIGHT_FINGER_PRE_OPEN)

        mujoco.mj_forward(self.model, self.data)
        sim_steps(self.model, self.data, steps=120, viewer=self.viewer, realtime=self.realtime)


class RightArmAdapter:
    def __init__(self, model, data, viewer=None, realtime: bool = False):
        self.model = model
        self.data = data
        self.viewer = viewer
        self.realtime = realtime

    def pick_and_place(self, task: PickPlaceTask) -> TaskResult:
        print(f"[RIGHT ADAPTER] pick_and_place: {task.task_id}")
        print("[RIGHT ADAPTER] cube_pos:", task.cube_pos)
        print("[RIGHT ADAPTER] config:", RIGHT_BEST_CONFIG["name"])

        right_rule.FIXED_CUBE_POS = np.asarray(task.cube_pos, dtype=float).copy()
        site_name = right_rule.choose_right_site(self.model)

        result = right_rule.run_trial(
            model=self.model,
            site_name=site_name,
            config=RIGHT_BEST_CONFIG,
            do_place=True,
            viewer=self.viewer,
            realtime=self.realtime,
            data=self.data,
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

    def reset_scene(self, cube_pos: np.ndarray) -> None:
        load_home(self.model, self.data)
        set_free_body_pos(self.model, self.data, "orange_cube", cube_pos)
        set_ctrl(self.model, self.data, "left_finger1_ctrl", self.finger_pre_open)
        set_ctrl(self.model, self.data, "right_finger1_ctrl", right_rule.RIGHT_FINGER_PRE_OPEN)
        set_ctrl(self.model, self.data, "lifter_ctrl", 0.0)
        mujoco.mj_forward(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()
        sim_steps(self.model, self.data, steps=700, viewer=self.viewer, realtime=self.realtime)

        # 放稳后再重置 cube，保证初始状态稳定。
        set_free_body_pos(self.model, self.data, "orange_cube", cube_pos)
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
            if self.realtime and self.viewer is not None:
                sleep_time = self.model.opt.timestep - (time.time() - start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def pick_and_place(self, task: PickPlaceTask) -> TaskResult:
        print(f"[LEFT ADAPTER] pick_and_place: {task.task_id}")
        print("[LEFT ADAPTER] cube_pos:", task.cube_pos)
        print("[LEFT ADAPTER] site_name:", self.site_name)
        print("[LEFT ADAPTER] pregrasp_offset:", self.pregrasp_offset)
        print("[LEFT ADAPTER] grasp_offset:", self.grasp_offset)

        self.reset_scene(task.cube_pos)

        cube_initial = get_body_pos(self.model, self.data, "orange_cube")
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
        cube_now = get_body_pos(self.model, self.data, "orange_cube")
        self.ik_move_to(cube_now + self.pregrasp_offset, "pregrasp", duration=2.0)

        # Grasp.
        cube_now = get_body_pos(self.model, self.data, "orange_cube")
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
            cube_now = get_body_pos(self.model, self.data, "orange_cube")
            max_cube_z = max(max_cube_z, float(cube_now[2]))
            if self.viewer is not None:
                self.viewer.sync()
            if self.realtime and self.viewer is not None:
                sleep_time = self.model.opt.timestep - (time.time() - start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        cube_after_lift = get_body_pos(self.model, self.data, "orange_cube")
        lift_delta = float(max_cube_z - cube_initial_z)
        final_lift_delta = float(cube_after_lift[2] - cube_initial_z)
        pick_success = bool(lift_delta > self.lift_success_delta_z)
        print(f"[LEFT ADAPTER] lift_delta={lift_delta:.4f}, pick_success={pick_success}")

        if pick_success:
            frame_pos = get_body_pos(self.model, self.data, "black_frame")
            self.ik_move_to(frame_pos + self.preplace_offset, "preplace", duration=2.0)

            frame_pos = get_body_pos(self.model, self.data, "black_frame")
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

        final_cube = get_body_pos(self.model, self.data, "orange_cube")
        frame_pos = get_body_pos(self.model, self.data, "black_frame")
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
# Planner
# ============================================================

class BimanualTaskPlanner:
    def __init__(self, model, data, viewer=None, realtime: bool = False):
        self.model = model
        self.data = data
        self.viewer = viewer
        self.realtime = realtime
        self.logger = TaskLogger(LOG_PATH)
        self.safety = SafetyManager(model, data, viewer, realtime)
        self.left_arm = LeftArmAdapter(model, data, viewer, realtime)
        self.right_arm = RightArmAdapter(model, data, viewer, realtime)

    def perceive(self, task: PickPlaceTask) -> Dict[str, np.ndarray]:
        # 展示版感知：任务定义给出 cube_pos，同时写入仿真场景。
        print(f"[PERCEIVE] task={task.task_id}, object={task.object_name}, target={task.target_name}")
        print("[PERCEIVE] planned cube_pos:", task.cube_pos)
        return {"cube_pos": task.cube_pos.copy()}

    def select_arm(self, cube_pos: np.ndarray) -> str:
        arm = "right" if float(cube_pos[1]) >= SELECT_RIGHT_IF_CUBE_Y_GE else "left"
        print(f"[SELECT_ARM] cube_y={cube_pos[1]:.3f}, threshold={SELECT_RIGHT_IF_CUBE_Y_GE:.3f} -> {arm}")
        return arm

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
        selected_arm = self.select_arm(obs["cube_pos"])
        self.logger.log(task, state, selected_arm, message="arm selected")

        state = PlannerState.PARK_INACTIVE_ARM
        self.safety.park_inactive_arm(selected_arm)
        self.logger.log(task, state, selected_arm, message="inactive arm parked / safe")

        state = PlannerState.EXECUTE_PICK_PLACE
        self.logger.log(task, state, selected_arm, message="start pick_and_place")
        if selected_arm == "right":
            result = self.right_arm.pick_and_place(task)
        else:
            result = self.left_arm.pick_and_place(task)
        self.logger.log(task, state, selected_arm, result=result, message="pick_and_place finished")

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


def print_presentation_header(tasks: List[PickPlaceTask]) -> None:
    print("=" * 100)
    print("OpenArm 双臂上层任务规划 Demo")
    print("=" * 100)
    print("核心演示：")
    print("  1. 单一 planner 接收任务序列，不再手动分开运行左右臂脚本。")
    print("  2. planner 根据 cube_y 自动选择 left/right arm。")
    print("  3. active arm 执行时，inactive arm 进入 park/hold-safe 状态，作为基础避让。")
    print("  4. 两个任务连续执行，目标是两条手臂各执行一次 pick-and-place。")
    print("")
    print("State machine:")
    print("  INIT -> PERCEIVE -> SELECT_ARM -> PARK_INACTIVE_ARM -> EXECUTE_PICK_PLACE -> VERIFY -> DONE/FAILED")
    print("")
    print(f"selector: right if cube_y >= {SELECT_RIGHT_IF_CUBE_Y_GE:.3f}, else left")
    print("XML_PATH:", XML_PATH)
    print("LOG_PATH:", LOG_PATH)
    print("")
    print("Tasks:")
    for task in tasks:
        arm = "right" if task.cube_pos[1] >= SELECT_RIGHT_IF_CUBE_Y_GE else "left"
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
    print("=" * 100)
    if success_count == len(results):
        print("结论：双臂任务规划、自动选臂、基础避让、连续执行已完成。")
    else:
        print("说明：上层 planner 与右臂分支已可演示；失败分支可作为后续 left TCP/BC 接入优化点。")


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="OpenArm bimanual task planner demo")
    parser.add_argument("--no-viewer", action="store_true", help="不打开 viewer，快速 headless 测试")
    parser.add_argument("--right-only", action="store_true", help="只运行右臂稳定展示任务")
    parser.add_argument("--no-hold-viewer", dest="hold_viewer", action="store_false", help="结束后不保持 viewer")
    parser.set_defaults(hold_viewer=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not XML_PATH.exists():
        raise FileNotFoundError(f"找不到 XML：{XML_PATH}")

    ensure_output_dir()

    raw_tasks = RIGHT_ONLY_TASKS if args.right_only else DEFAULT_TASKS
    tasks = make_tasks(raw_tasks)

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    if args.no_viewer:
        planner = BimanualTaskPlanner(model, data, viewer=None, realtime=False)
        planner.run(tasks)
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        print("viewer 已打开。2 秒后开始双臂任务规划 demo。")
        time.sleep(2.0)

        planner = BimanualTaskPlanner(model, data, viewer=viewer, realtime=True)
        planner.run(tasks)

        if args.hold_viewer:
            print("viewer 保持运行。关闭 viewer 或 Ctrl+C 结束。")
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()

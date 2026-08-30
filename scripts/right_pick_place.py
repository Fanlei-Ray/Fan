from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# 路径
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
XML_PATH = ROOT / "v2" / "demo.xml"


# ============================================================
# 左臂 IK 配置
# ============================================================

LEFT_SITE_NAME = "left_gripper_tcp"

LEFT_JOINT_NAMES = [
    "openarm_left_joint1",
    "openarm_left_joint2",
    "openarm_left_joint3",
    "openarm_left_joint4",
    "openarm_left_joint5",
    "openarm_left_joint6",
    "openarm_left_joint7",
]

LEFT_ACTUATOR_NAMES = [
    "left_joint1_ctrl",
    "left_joint2_ctrl",
    "left_joint3_ctrl",
    "left_joint4_ctrl",
    "left_joint5_ctrl",
    "left_joint6_ctrl",
    "left_joint7_ctrl",
]


# ============================================================
# 任务参数
# ============================================================

PICK_BODY_NAME = "orange_cube"
PLACE_BODY_NAME = "black_frame"

LEFT_FINGER_OPEN = 0.49
LEFT_FINGER_PRE_OPEN = 0.445
LEFT_FINGER_CLOSE = 0.0

LIFTER_HOME = 0.0
LIFTER_UP = 0.11

LIFT_SUCCESS_DELTA_Z = 0.015

# 你跑通后的 TCP 偏移参数
PICK_PREGRASP_OFFSET = np.array([-0.005, 0.00, 0.10])
PICK_GRASP_OFFSET = np.array([-0.010, 0.00, -0.005])

PLACE_PREPLACE_OFFSET = np.array([0.0, 0.0, 0.14])
PLACE_RELEASE_OFFSET = np.array([0.0, 0.0, 0.08])


# ============================================================
# 基础工具函数
# ============================================================

def get_id(model, obj_type, name):
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id == -1:
        raise ValueError(f"找不到对象：{name}")
    return obj_id


def joint_id(model, name):
    return get_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def actuator_id(model, name):
    return get_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def body_id(model, name):
    return get_id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def site_id(model, name):
    return get_id(model, mujoco.mjtObj.mjOBJ_SITE, name)


def get_body_pos(model, data, name):
    bid = body_id(model, name)
    return data.xpos[bid].copy()


def get_site_pos(model, data, name):
    sid = site_id(model, name)
    return data.site_xpos[sid].copy()


def print_actuators(model):
    print("\n========== Actuators ==========")
    for aid in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        low, high = model.actuator_ctrlrange[aid]
        print(f"{aid:02d}  {name:30s}  ctrlrange=[{low:.4f}, {high:.4f}]")
    print("================================\n")


def print_left_ctrl_values(ctrl_targets):
    print("\n========== 左臂 IK ctrl 目标 ==========")
    for name in LEFT_ACTUATOR_NAMES:
        print(f'"{name}": {ctrl_targets[name]:.6f},')
    print("=====================================\n")


# ============================================================
# 初始化
# ============================================================

def load_home(model, data):
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        print("已加载 home keyframe。")
    else:
        print("没有找到 home keyframe，使用默认姿态。")

    # 把 actuator ctrl 同步到当前 qpos，防止机械臂垂下去
    for aid in range(model.nu):
        jid = model.actuator_trnid[aid, 0]
        if jid < 0:
            continue

        qaddr = model.jnt_qposadr[jid]
        low, high = model.actuator_ctrlrange[aid]
        data.ctrl[aid] = np.clip(data.qpos[qaddr], low, high)

    mujoco.mj_forward(model, data)


# ============================================================
# 仿真运动函数
# ============================================================

def hold(model, data, viewer, duration=0.5):
    dt = model.opt.timestep
    steps = max(1, int(duration / dt))

    for _ in range(steps):
        if not viewer.is_running():
            return

        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()

        sleep_time = dt - (time.time() - step_start)
        if sleep_time > 0:
            time.sleep(sleep_time)


def move_to(model, data, viewer, targets, duration=1.5):
    start_ctrl = data.ctrl.copy()
    goal_ctrl = data.ctrl.copy()

    for name, value in targets.items():
        aid = actuator_id(model, name)
        low, high = model.actuator_ctrlrange[aid]
        goal_ctrl[aid] = np.clip(value, low, high)

    dt = model.opt.timestep
    steps = max(1, int(duration / dt))

    for i in range(steps):
        if not viewer.is_running():
            return False

        alpha = (i + 1) / steps
        alpha = 3 * alpha ** 2 - 2 * alpha ** 3

        data.ctrl[:] = (1 - alpha) * start_ctrl + alpha * goal_ctrl

        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()

        sleep_time = dt - (time.time() - step_start)
        if sleep_time > 0:
            time.sleep(sleep_time)

    return True


def open_gripper(model, data, viewer, value=LEFT_FINGER_PRE_OPEN, duration=1.0):
    print(f"张开左夹爪到 {value:.3f}")
    move_to(
        model,
        data,
        viewer,
        {"left_finger1_ctrl": value},
        duration=duration,
    )
    hold(model, data, viewer, duration=0.5)


def close_gripper_slowly(
    model,
    data,
    viewer,
    start_value=LEFT_FINGER_PRE_OPEN,
    end_value=LEFT_FINGER_CLOSE,
    duration=2.5,
):
    aid = actuator_id(model, "left_finger1_ctrl")
    low, high = model.actuator_ctrlrange[aid]

    start_value = float(np.clip(start_value, low, high))
    end_value = float(np.clip(end_value, low, high))

    print(f"自动闭合夹爪：{start_value:.4f} -> {end_value:.4f}")

    dt = model.opt.timestep
    steps = max(1, int(duration / dt))

    for i in range(steps):
        if not viewer.is_running():
            return False

        alpha = (i + 1) / steps
        alpha = 3 * alpha ** 2 - 2 * alpha ** 3

        data.ctrl[aid] = (1 - alpha) * start_value + alpha * end_value

        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()

        sleep_time = dt - (time.time() - step_start)
        if sleep_time > 0:
            time.sleep(sleep_time)

    return True


def move_lifter(model, data, viewer, value, duration=1.5):
    print(f"移动 lifter 到 {value:.3f}")
    move_to(
        model,
        data,
        viewer,
        {"lifter_ctrl": value},
        duration=duration,
    )
    hold(model, data, viewer, duration=0.5)


def is_body_lifted(model, data, body_name, initial_z, delta_z=LIFT_SUCCESS_DELTA_Z):
    pos = get_body_pos(model, data, body_name)
    current_z = pos[2]
    delta = current_z - initial_z

    print(
        f"{body_name} 高度检测：initial_z={initial_z:.4f}, "
        f"current_z={current_z:.4f}, delta={delta:.4f}"
    )

    return delta > delta_z


# ============================================================
# IK 求解
# ============================================================

def get_left_joint_info(model):
    qpos_addrs = []
    dof_addrs = []
    joint_ranges = []

    for name in LEFT_JOINT_NAMES:
        jid = joint_id(model, name)
        qpos_addrs.append(model.jnt_qposadr[jid])
        dof_addrs.append(model.jnt_dofadr[jid])
        joint_ranges.append(model.jnt_range[jid].copy())

    return qpos_addrs, dof_addrs, np.array(joint_ranges)


def make_ik_data(model, source_data):
    ik_data = mujoco.MjData(model)

    ik_data.qpos[:] = source_data.qpos[:]
    ik_data.qvel[:] = source_data.qvel[:]
    ik_data.ctrl[:] = source_data.ctrl[:]

    mujoco.mj_forward(model, ik_data)
    return ik_data


def solve_left_arm_ik(
    model,
    source_data,
    target_pos,
    site_name=LEFT_SITE_NAME,
    max_iters=200,
    tolerance=1e-3,
    step_size=0.6,
    damping=1e-3,
):
    ik_data = make_ik_data(model, source_data)

    sid = site_id(model, site_name)
    qpos_addrs, dof_addrs, joint_ranges = get_left_joint_info(model)

    target_pos = np.asarray(target_pos, dtype=float)

    for it in range(max_iters):
        mujoco.mj_forward(model, ik_data)

        current_pos = ik_data.site_xpos[sid].copy()
        error = target_pos - current_pos
        err_norm = np.linalg.norm(error)

        if err_norm < tolerance:
            print(f"IK 成功：iter={it}, error={err_norm:.6f}")
            break

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, ik_data, jacp, jacr, sid)

        J = jacp[:, dof_addrs]

        A = J @ J.T + damping * np.eye(3)
        dq = J.T @ np.linalg.solve(A, error)

        dq = step_size * dq
        dq = np.clip(dq, -0.05, 0.05)

        for i, qaddr in enumerate(qpos_addrs):
            ik_data.qpos[qaddr] += dq[i]

            low, high = joint_ranges[i]
            ik_data.qpos[qaddr] = np.clip(ik_data.qpos[qaddr], low, high)

    mujoco.mj_forward(model, ik_data)

    final_pos = ik_data.site_xpos[sid].copy()
    final_error = np.linalg.norm(target_pos - final_pos)

    success = final_error < 0.01

    if success:
        print(f"IK 最终误差：{final_error:.6f}")
    else:
        print(f"IK 未完全收敛：final_error={final_error:.6f}")
        print(f"target_pos={target_pos}")
        print(f"final_pos ={final_pos}")

    ctrl_targets = {}

    for act_name, joint_name in zip(LEFT_ACTUATOR_NAMES, LEFT_JOINT_NAMES):
        aid = actuator_id(model, act_name)
        jid = joint_id(model, joint_name)
        qaddr = model.jnt_qposadr[jid]

        low, high = model.actuator_ctrlrange[aid]
        ctrl_targets[act_name] = float(np.clip(ik_data.qpos[qaddr], low, high))

    print_left_ctrl_values(ctrl_targets)

    return success, ctrl_targets


def ik_move_left_tcp_to(model, data, viewer, target_pos, label, duration=2.0):
    print(f"\nIK 移动：{label}")
    print(f"目标点：{target_pos}")

    success, ctrl_targets = solve_left_arm_ik(model, data, target_pos)

    if not success:
        print(f"错误：{label} 的 IK 误差太大，停止这一步。")
        return False

    move_to(model, data, viewer, ctrl_targets, duration=duration)
    hold(model, data, viewer, duration=0.5)

    actual_pos = get_site_pos(model, data, LEFT_SITE_NAME)
    print(f"移动后 {LEFT_SITE_NAME} 位置：{actual_pos}")

    return True


def ik_move_to_body_offset(model, data, viewer, body_name, offset, label, duration=2.0):
    body_pos = get_body_pos(model, data, body_name)
    target_pos = body_pos + offset

    print(f"\n目标 body：{body_name}")
    print(f"{body_name} 当前世界坐标：{body_pos}")
    print(f"offset：{offset}")
    print(f"最终 IK 目标：{target_pos}")

    return ik_move_left_tcp_to(
        model,
        data,
        viewer,
        target_pos,
        label=label,
        duration=duration,
    )


# ============================================================
# 高层任务函数
# ============================================================

def stabilize_object(model, data, viewer, body_name, duration=1.0):
    print(f"\n让 {body_name} 稳定 {duration:.1f} 秒。")
    hold(model, data, viewer, duration=duration)

    pos = get_body_pos(model, data, body_name)
    print(f"{body_name} 稳定后位置：{pos}")

    return pos


def pick_object(model, data, viewer, body_name):
    print("\n==================================================")
    print(f"开始抓取：{body_name}")
    print("==================================================")

    initial_pos = stabilize_object(model, data, viewer, body_name, duration=1.0)
    initial_z = initial_pos[2]

    open_gripper(model, data, viewer, value=LEFT_FINGER_PRE_OPEN, duration=1.0)

    ok = ik_move_to_body_offset(
        model,
        data,
        viewer,
        body_name,
        PICK_PREGRASP_OFFSET,
        label=f"{body_name} 上方 pregrasp",
        duration=2.0,
    )
    if not ok:
        return False

    ok = ik_move_to_body_offset(
        model,
        data,
        viewer,
        body_name,
        PICK_GRASP_OFFSET,
        label=f"{body_name} 抓取位置 grasp",
        duration=3.0,
    )
    if not ok:
        return False

    close_gripper_slowly(
        model,
        data,
        viewer,
        start_value=LEFT_FINGER_PRE_OPEN,
        end_value=LEFT_FINGER_CLOSE,
        duration=2.5,
    )
    hold(model, data, viewer, duration=1.0)

    move_lifter(model, data, viewer, value=LIFTER_UP, duration=1.5)
    hold(model, data, viewer, duration=1.0)

    success = is_body_lifted(model, data, body_name, initial_z)

    if success:
        print(f"抓取成功：{body_name} 已被抬起。")
        return True

    print(f"抓取失败：{body_name} 没有明显被抬起。")
    return False


def place_object(model, data, viewer, target_body_name):
    print("\n==================================================")
    print(f"开始放置到：{target_body_name}")
    print("==================================================")

    ok = ik_move_to_body_offset(
        model,
        data,
        viewer,
        target_body_name,
        PLACE_PREPLACE_OFFSET,
        label=f"{target_body_name} 上方 preplace",
        duration=2.0,
    )
    if not ok:
        return False

    ok = ik_move_to_body_offset(
        model,
        data,
        viewer,
        target_body_name,
        PLACE_RELEASE_OFFSET,
        label=f"{target_body_name} 释放位置 release",
        duration=1.5,
    )
    if not ok:
        return False

    open_gripper(model, data, viewer, value=LEFT_FINGER_OPEN, duration=1.0)
    hold(model, data, viewer, duration=1.0)

    print(f"释放完成：{target_body_name}")
    return True


def run_pick_and_place_task(model, data, viewer):
    print("\n##################################################")
    print("任务：pick orange_cube -> place black_frame")
    print("##################################################")

    picked = pick_object(model, data, viewer, PICK_BODY_NAME)

    if not picked:
        print("\n任务失败：抓取阶段失败。")
        return False

    placed = place_object(model, data, viewer, PLACE_BODY_NAME)

    if not placed:
        print("\n任务失败：放置阶段失败。")
        return False

    print("\n任务成功：已完成 pick-and-place。")
    return True


# ============================================================
# 主入口
# ============================================================

def main():
    print(f"加载模型：{XML_PATH}")

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    load_home(model, data)
    print_actuators(model)

    cube_pos = get_body_pos(model, data, PICK_BODY_NAME)
    frame_pos = get_body_pos(model, data, PLACE_BODY_NAME)
    tcp_pos = get_site_pos(model, data, LEFT_SITE_NAME)

    print(f"{PICK_BODY_NAME} 初始位置：{cube_pos}")
    print(f"{PLACE_BODY_NAME} 初始位置：{frame_pos}")
    print(f"{LEFT_SITE_NAME} 初始位置：{tcp_pos}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()

        print("viewer 已打开。")
        print("3 秒后开始执行任务式 pick-and-place。")
        time.sleep(3)

        run_pick_and_place_task(model, data, viewer)

        print("\n流程结束，viewer 保持运行。")

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
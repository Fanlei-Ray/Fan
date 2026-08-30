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
# 夹爪和任务参数
# ============================================================

LEFT_FINGER_OPEN = 0.49
LEFT_FINGER_PRE_OPEN = 0.445
LEFT_FINGER_CLOSE = 0.0

LIFTER_UP = 0.11

# 抓取阶段抬一点手腕，避免夹爪先碰到方块把方块推走
# 如果发现方向反了，就改成 -0.08
LEFT_JOINT6_GRASP_BIAS = 0.0

# 判断方块是否被抬起：当前 z 比初始 z 高多少算成功
LIFT_SUCCESS_DELTA_Z = 0.015


# ============================================================
# IK 目标点偏移
# ============================================================

CUBE_PREGRASP_OFFSET = np.array([-0.005, 0.00, 0.10])
CUBE_GRASP_OFFSET = np.array([-0.010, 0.00, -0.005])

FRAME_PREPLACE_OFFSET = np.array([0.0, 0.0, 0.14])
FRAME_PLACE_OFFSET = np.array([0.0, 0.0, 0.08])


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


def set_ctrl(model, data, name, value):
    aid = actuator_id(model, name)
    low, high = model.actuator_ctrlrange[aid]
    data.ctrl[aid] = np.clip(value, low, high)


def print_left_ctrl_values(model, ctrl_targets):
    print("\n========== 左臂 IK ctrl 目标 ==========")
    for name in LEFT_ACTUATOR_NAMES:
        print(f'"{name}": {ctrl_targets[name]:.6f},')
    print("=====================================\n")


# ============================================================
# 初始化
# ============================================================

def load_home(model, data):
    """
    加载 XML 里的 home keyframe。
    然后把 position actuator 的 ctrl 同步到当前 qpos，
    防止机械臂一启动就垂下去。
    """
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        print("已加载 home keyframe。")
    else:
        print("没有找到 home keyframe，使用默认姿态。")

    for aid in range(model.nu):
        jid = model.actuator_trnid[aid, 0]
        if jid < 0:
            continue

        qaddr = model.jnt_qposadr[jid]
        low, high = model.actuator_ctrlrange[aid]
        data.ctrl[aid] = np.clip(data.qpos[qaddr], low, high)

    mujoco.mj_forward(model, data)


def print_actuators(model):
    print("\n========== Actuators ==========")
    for aid in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        low, high = model.actuator_ctrlrange[aid]
        print(f"{aid:02d}  {name:30s}  ctrlrange=[{low:.4f}, {high:.4f}]")
    print("================================\n")


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
    """
    平滑移动到目标 ctrl。
    targets 是 actuator_name -> value 的字典。
    """
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
            return

        alpha = (i + 1) / steps
        alpha = 3 * alpha ** 2 - 2 * alpha ** 3

        data.ctrl[:] = (1 - alpha) * start_ctrl + alpha * goal_ctrl

        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()

        sleep_time = dt - (time.time() - step_start)
        if sleep_time > 0:
            time.sleep(sleep_time)


def close_gripper_slowly(
    model,
    data,
    viewer,
    actuator_name="left_finger1_ctrl",
    start_value=LEFT_FINGER_PRE_OPEN,
    end_value=LEFT_FINGER_CLOSE,
    duration=1.5,
):
    """
    自动夹爪闭合：从张开值慢慢闭合到关闭值。
    """
    aid = actuator_id(model, actuator_name)
    low, high = model.actuator_ctrlrange[aid]

    start_value = float(np.clip(start_value, low, high))
    end_value = float(np.clip(end_value, low, high))

    print(f"自动闭合夹爪：{start_value:.4f} -> {end_value:.4f}")

    dt = model.opt.timestep
    steps = max(1, int(duration / dt))

    for i in range(steps):
        if not viewer.is_running():
            return

        alpha = (i + 1) / steps
        alpha = 3 * alpha ** 2 - 2 * alpha ** 3

        data.ctrl[aid] = (1 - alpha) * start_value + alpha * end_value

        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()

        sleep_time = dt - (time.time() - step_start)
        if sleep_time > 0:
            time.sleep(sleep_time)


def is_cube_lifted(model, data, initial_cube_z, delta_z=LIFT_SUCCESS_DELTA_Z):
    cube_pos = get_body_pos(model, data, "orange_cube")
    current_z = cube_pos[2]
    delta = current_z - initial_cube_z

    print(
        f"方块高度检测：initial_z={initial_cube_z:.4f}, "
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
    """
    新建一个 IK 专用 data，复制当前仿真状态。
    IK 会修改 ik_data.qpos，不直接破坏真实仿真 data。
    """
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
    """
    只做位置 IK，不控制末端姿态。
    返回值：
        success, ctrl_targets
    """
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

    print_left_ctrl_values(model, ctrl_targets)

    return success, ctrl_targets


def ik_move_left_ee_to(
    model,
    data,
    viewer,
    target_pos,
    label,
    duration=2.0,
    joint6_bias=0.0,
):
    """
    用 IK 计算左臂目标 ctrl，然后平滑移动过去。
    joint6_bias 用来在抓取阶段抬一点手腕，避免夹爪先碰到方块。
    """
    print(f"\nIK 移动：{label}")
    print(f"目标点：{target_pos}")

    success, ctrl_targets = solve_left_arm_ik(model, data, target_pos)

    if joint6_bias != 0.0:
        aid = actuator_id(model, "left_joint6_ctrl")
        low, high = model.actuator_ctrlrange[aid]

        old_value = ctrl_targets["left_joint6_ctrl"]
        new_value = float(np.clip(old_value + joint6_bias, low, high))
        ctrl_targets["left_joint6_ctrl"] = new_value

        print(
            f"给 left_joint6_ctrl 加抬腕偏置："
            f"{old_value:.4f} -> {new_value:.4f}"
        )

    if not success:
        print(f"错误：{label} 的 IK 误差太大，停止这一步。")
        return False

    move_to(model, data, viewer, ctrl_targets, duration=duration)
    hold(model, data, viewer, duration=0.5)

    actual_pos = get_site_pos(model, data, LEFT_SITE_NAME)
    print(f"移动后 {LEFT_SITE_NAME} 位置：{actual_pos}")

    return success


# ============================================================
# 主流程
# ============================================================

def main():
    print(f"加载模型：{XML_PATH}")

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    load_home(model, data)
    print_actuators(model)

    cube_initial_pos = get_body_pos(model, data, "orange_cube")
    frame_initial_pos = get_body_pos(model, data, "black_frame")
    ee_initial_pos = get_site_pos(model, data, LEFT_SITE_NAME)

    cube_initial_z = cube_initial_pos[2]

    print(f"orange_cube 初始位置：{cube_initial_pos}")
    print(f"black_frame 初始位置：{frame_initial_pos}")
    print(f"{LEFT_SITE_NAME} 初始位置：{ee_initial_pos}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()

        print("viewer 已打开。")
        print("3 秒后开始执行 IK 版本左臂 pick-and-place。")
        time.sleep(3)

        # 先让方块在桌面上稳定下来，再记录真正的初始高度
        print("让方块稳定 1 秒，然后重新记录初始高度。")
        hold(model, data, viewer, duration=1.0)

        cube_initial_pos = get_body_pos(model, data, "orange_cube")
        cube_initial_z = cube_initial_pos[2]

        print(f"方块稳定后位置：{cube_initial_pos}")
        print(f"方块稳定后初始高度 z：{cube_initial_z:.4f}")

        # ------------------------------------------------------------
        # 1. 张开夹爪
        # ------------------------------------------------------------
        print("\n1. 张开左夹爪")
        move_to(
            model,
            data,
            viewer,
            {"left_finger1_ctrl": LEFT_FINGER_PRE_OPEN},
            duration=1.0,
        )
        hold(model, data, viewer, duration=0.5)

        # ------------------------------------------------------------
        # 2. IK 到方块上方
        # ------------------------------------------------------------
        cube_pos = get_body_pos(model, data, "orange_cube")

        target_pregrasp = cube_pos + CUBE_PREGRASP_OFFSET
        ik_move_left_ee_to(
            model,
            data,
            viewer,
            target_pregrasp,
            label="方块上方 pregrasp",
            duration=2.0,
            joint6_bias=LEFT_JOINT6_GRASP_BIAS,
        )

        # ------------------------------------------------------------
        # 3. IK 下探到抓取位置
        # ------------------------------------------------------------
        cube_pos = get_body_pos(model, data, "orange_cube")

        target_grasp = cube_pos + CUBE_GRASP_OFFSET
        ik_move_left_ee_to(
            model,
            data,
            viewer,
            target_grasp,
            label="方块抓取位置 grasp",
            duration=3.0,
            joint6_bias=LEFT_JOINT6_GRASP_BIAS,
        )
        hold(model, data, viewer, duration=1.0)

        # ------------------------------------------------------------
        # 4. 自动闭合夹爪
        # ------------------------------------------------------------
        print("\n4. 自动慢慢闭合左夹爪")
        close_gripper_slowly(
            model,
            data,
            viewer,
            actuator_name="left_finger1_ctrl",
            start_value=LEFT_FINGER_PRE_OPEN,
            end_value=LEFT_FINGER_CLOSE,
            duration=2.5,
        )
        hold(model, data, viewer, duration=1.0)

        # ------------------------------------------------------------
        # 5. 抬升 lifter
        # ------------------------------------------------------------
        print("\n5. lifter 抬高")
        move_to(
            model,
            data,
            viewer,
            {"lifter_ctrl": LIFTER_UP},
            duration=1.5,
        )
        hold(model, data, viewer, duration=1.0)

        # ------------------------------------------------------------
        # 6. 检测是否抓取成功
        # ------------------------------------------------------------
        print("\n6. 检测是否抓取成功")
        success = is_cube_lifted(model, data, cube_initial_z)

        if not success:
            print("抓取失败：方块没有明显被抬起。")
            print("流程停止。你需要微调 CUBE_GRASP_OFFSET、CUBE_PREGRASP_OFFSET、夹爪开合或 LEFT_JOINT6_GRASP_BIAS。")

            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)

            return

        print("抓取成功：继续执行放置。")

        # ------------------------------------------------------------
        # 7. IK 移动到黑框上方
        # ------------------------------------------------------------
        frame_pos = get_body_pos(model, data, "black_frame")

        target_preplace = frame_pos + FRAME_PREPLACE_OFFSET
        ik_move_left_ee_to(
            model,
            data,
            viewer,
            target_preplace,
            label="黑框上方 preplace",
            duration=2.0,
        )

        # ------------------------------------------------------------
        # 8. IK 下降到释放位置
        # ------------------------------------------------------------
        frame_pos = get_body_pos(model, data, "black_frame")

        target_place = frame_pos + FRAME_PLACE_OFFSET
        ik_move_left_ee_to(
            model,
            data,
            viewer,
            target_place,
            label="黑框释放位置 place",
            duration=1.5,
        )

        # ------------------------------------------------------------
        # 9. 张开夹爪释放
        # ------------------------------------------------------------
        print("\n9. 张开左夹爪释放")
        move_to(
            model,
            data,
            viewer,
            {"left_finger1_ctrl": LEFT_FINGER_OPEN},
            duration=1.0,
        )
        hold(model, data, viewer, duration=1.0)

        # ------------------------------------------------------------
        # 10. 结束保持
        # ------------------------------------------------------------
        print("\nIK 版本左臂抓取 + 放置流程执行完毕。viewer 保持运行。")

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
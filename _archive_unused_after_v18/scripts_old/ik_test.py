from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
XML_PATH = ROOT / "v2" / "demo.xml"


LEFT_SITE_NAME = "left_ee_control_point"

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


def get_id(model, obj_type, name):
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id == -1:
        raise ValueError(f"找不到对象：{name}")
    return obj_id


def joint_id(model, name):
    return get_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def actuator_id(model, name):
    return get_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def site_id(model, name):
    return get_id(model, mujoco.mjtObj.mjOBJ_SITE, name)


def body_id(model, name):
    return get_id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def load_home(model, data):
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        print("已加载 home keyframe。")
    else:
        print("没有找到 home keyframe。")

    for aid in range(model.nu):
        jid = model.actuator_trnid[aid, 0]
        if jid < 0:
            continue

        qpos_addr = model.jnt_qposadr[jid]
        low, high = model.actuator_ctrlrange[aid]
        data.ctrl[aid] = np.clip(data.qpos[qpos_addr], low, high)

    mujoco.mj_forward(model, data)


def get_body_pos(model, data, name):
    bid = body_id(model, name)
    return data.xpos[bid].copy()


def get_site_pos(model, data, name):
    sid = site_id(model, name)
    return data.site_xpos[sid].copy()


def get_left_joint_info(model):
    """
    返回左臂 7 个关节的 joint_id、qpos_addr、dof_addr。
    IK 里面需要用 dof_addr 去取 Jacobian 的列。
    """
    joint_ids = []
    qpos_addrs = []
    dof_addrs = []
    ranges = []

    for name in LEFT_JOINT_NAMES:
        jid = joint_id(model, name)
        joint_ids.append(jid)
        qpos_addrs.append(model.jnt_qposadr[jid])
        dof_addrs.append(model.jnt_dofadr[jid])
        ranges.append(model.jnt_range[jid].copy())

    return joint_ids, qpos_addrs, dof_addrs, np.array(ranges)


def sync_left_ctrl_to_qpos(model, data):
    """
    把左臂 actuator 的 ctrl 设置成当前 qpos。
    IK 算完后，需要让位置控制器追踪这个姿态。
    """
    for act_name, joint_name in zip(LEFT_ACTUATOR_NAMES, LEFT_JOINT_NAMES):
        aid = actuator_id(model, act_name)
        jid = joint_id(model, joint_name)

        qpos_addr = model.jnt_qposadr[jid]
        low, high = model.actuator_ctrlrange[aid]

        data.ctrl[aid] = np.clip(data.qpos[qpos_addr], low, high)


def solve_left_arm_ik(
    model,
    data,
    target_pos,
    site_name=LEFT_SITE_NAME,
    max_iters=200,
    tolerance=1e-3,
    step_size=0.6,
    damping=1e-3,
):
    """
    只做位置 IK，不管末端姿态。
    目标：让 left_ee_control_point 移动到 target_pos。

    使用阻尼最小二乘：
        dq = J.T @ inv(J @ J.T + damping * I) @ error
    """
    sid = site_id(model, site_name)
    _, qpos_addrs, dof_addrs, joint_ranges = get_left_joint_info(model)

    target_pos = np.asarray(target_pos, dtype=float)

    for it in range(max_iters):
        mujoco.mj_forward(model, data)

        current_pos = data.site_xpos[sid].copy()
        error = target_pos - current_pos
        err_norm = np.linalg.norm(error)

        if err_norm < tolerance:
            print(f"IK 成功：iter={it}, error={err_norm:.6f}")
            return True

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, sid)

        # 只取左臂 7 个关节对应的 Jacobian 列
        J = jacp[:, dof_addrs]

        # 阻尼最小二乘
        A = J @ J.T + damping * np.eye(3)
        dq = J.T @ np.linalg.solve(A, error)

        # 限制每次迭代不要走太猛
        dq = step_size * dq
        max_step = 0.05
        dq = np.clip(dq, -max_step, max_step)

        for i, qaddr in enumerate(qpos_addrs):
            data.qpos[qaddr] += dq[i]

            low, high = joint_ranges[i]
            data.qpos[qaddr] = np.clip(data.qpos[qaddr], low, high)

    mujoco.mj_forward(model, data)

    final_pos = data.site_xpos[sid].copy()
    final_error = np.linalg.norm(target_pos - final_pos)

    print(f"IK 未完全收敛：final_error={final_error:.6f}")
    print(f"target_pos={target_pos}")
    print(f"final_pos ={final_pos}")

    return False


def move_to_current_left_qpos(model, data, viewer, duration=2.0):
    """
    IK 直接改了 qpos。
    这个函数把当前 IK qpos 作为 actuator ctrl，然后让机器人平滑运动过去。
    """
    goal_ctrl = data.ctrl.copy()

    for act_name, joint_name in zip(LEFT_ACTUATOR_NAMES, LEFT_JOINT_NAMES):
        aid = actuator_id(model, act_name)
        jid = joint_id(model, joint_name)

        qaddr = model.jnt_qposadr[jid]
        low, high = model.actuator_ctrlrange[aid]
        goal_ctrl[aid] = np.clip(data.qpos[qaddr], low, high)

    # 为了平滑运动，先把 qpos 恢复到当前实际状态的 ctrl 附近
    # 这里使用 ctrl 插值，而不是直接跳。
    start_ctrl = data.ctrl.copy()

    dt = model.opt.timestep
    steps = max(1, int(duration / dt))

    for i in range(steps):
        if not viewer.is_running():
            return

        alpha = (i + 1) / steps
        alpha = 3 * alpha**2 - 2 * alpha**3

        data.ctrl[:] = (1 - alpha) * start_ctrl + alpha * goal_ctrl

        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()

        sleep_time = dt - (time.time() - step_start)
        if sleep_time > 0:
            time.sleep(sleep_time)


def hold(model, data, viewer, duration=1.0):
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


def print_left_ctrl_values(model, data):
    print("\n========== IK 算出来的左臂 ctrl ==========")

    for act_name, joint_name in zip(LEFT_ACTUATOR_NAMES, LEFT_JOINT_NAMES):
        aid = actuator_id(model, act_name)
        jid = joint_id(model, joint_name)

        qaddr = model.jnt_qposadr[jid]
        value = data.qpos[qaddr]

        low, high = model.actuator_ctrlrange[aid]
        value = np.clip(value, low, high)

        print(f'"{act_name}": {value:.6f},')

    print("=========================================\n")


def main():
    print(f"加载模型：{XML_PATH}")

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    load_home(model, data)

    cube_pos = get_body_pos(model, data, "orange_cube")
    frame_pos = get_body_pos(model, data, "black_frame")
    ee_pos = get_site_pos(model, data, LEFT_SITE_NAME)

    print(f"orange_cube pos = {cube_pos}")
    print(f"black_frame pos = {frame_pos}")
    print(f"{LEFT_SITE_NAME} pos = {ee_pos}")

    # 先测试：让 left_ee_control_point 移动到方块附近上方。
    #
    # 注意：
    # left_ee_control_point 不是两个手指中间点，而是末端控制点。
    # 所以这个 target 不一定就是夹爪中心，后面还要根据实际画面微调 offset。
    target_above_cube = cube_pos.copy()
    target_above_cube[0] = cube_pos[0] - 0.04
    target_above_cube[1] = cube_pos[1] + 0.02
    target_above_cube[2] = cube_pos[2] + 0.10

    print(f"IK target_above_cube = {target_above_cube}")

    success = solve_left_arm_ik(model, data, target_above_cube)

    print_left_ctrl_values(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()

        print("viewer 已打开。")
        print("3 秒后移动到 IK 算出的姿态。")
        time.sleep(3)

        sync_left_ctrl_to_qpos(model, data)
        hold(model, data, viewer, duration=3.0)

        print("IK 测试结束。viewer 保持运行。")

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
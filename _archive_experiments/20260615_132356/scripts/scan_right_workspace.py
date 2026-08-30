from pathlib import Path

import numpy as np
import mujoco


ROOT = Path(__file__).resolve().parents[1]
XML_PATH = ROOT / "v2" / "demo.xml"

RIGHT_SITE_NAME = "right_gripper_tcp"

RIGHT_JOINT_NAMES = [
    "openarm_right_joint1",
    "openarm_right_joint2",
    "openarm_right_joint3",
    "openarm_right_joint4",
    "openarm_right_joint5",
    "openarm_right_joint6",
    "openarm_right_joint7",
]

RIGHT_ACTUATOR_NAMES = [
    "right_joint1_ctrl",
    "right_joint2_ctrl",
    "right_joint3_ctrl",
    "right_joint4_ctrl",
    "right_joint5_ctrl",
    "right_joint6_ctrl",
    "right_joint7_ctrl",
]

RIGHT_FINGER_ACT = "right_finger1_ctrl"
LIFTER_ACT = "lifter_ctrl"

CUBE_BODY = "orange_cube"
FRAME_BODY = "black_frame"

# 右夹爪 ctrlrange 是 [-0.7854, 0.0]
# 右夹爪方向和左夹爪相反：负数张开，0 闭合
RIGHT_FINGER_OPEN = -0.49
RIGHT_FINGER_PRE_OPEN = -0.445
RIGHT_FINGER_CLOSE = 0.0

LIFTER_HOME = 0.0
LIFTER_UP = 0.11

PICK_PREGRASP_OFFSET = np.array([-0.005, 0.0, 0.10], dtype=np.float64)
PICK_GRASP_OFFSET = np.array([-0.010, 0.0, -0.005], dtype=np.float64)

PLACE_PREPLACE_OFFSET = np.array([0.0, 0.0, 0.14], dtype=np.float64)
PLACE_RELEASE_OFFSET = np.array([0.0, 0.0, 0.08], dtype=np.float64)

X_VALUES = np.round(np.linspace(0.46, 0.59, 8), 3)
Y_VALUES = np.round(np.linspace(-0.10, 0.10, 9), 3)

IK_ITERS = 120
IK_TOL = 0.004
IK_DAMPING = 1e-3

MOVE_STEPS = 90
GRASP_MOVE_STEPS = 100
PLACE_MOVE_STEPS = 100

SETTLE_STEPS = 80
HOLD_STEPS = 80
CLOSE_STEPS = 80
OPEN_STEPS = 60
LIFT_STEPS = 90


def name2id(model, obj_type, name):
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise RuntimeError(f"找不到 {name}")
    return obj_id


def get_act_id(model, name):
    return name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def get_body_id(model, name):
    return name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def get_site_id(model, name):
    return name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)


def get_joint_id(model, name):
    return name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def set_ctrl(model, data, act_name, value):
    act_id = get_act_id(model, act_name)
    low, high = model.actuator_ctrlrange[act_id]
    data.ctrl[act_id] = np.clip(value, low, high)


def step_n(model, data, n):
    for _ in range(n):
        mujoco.mj_step(model, data)


def load_home(model, data):
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)

    mujoco.mj_forward(model, data)

    # 把 position actuator 的 ctrl 同步到当前 qpos，避免一开始乱跳
    for act_name in RIGHT_ACTUATOR_NAMES + [RIGHT_FINGER_ACT, LIFTER_ACT]:
        act_id = get_act_id(model, act_name)
        joint_id = int(model.actuator_trnid[act_id, 0])
        if joint_id >= 0:
            qadr = int(model.jnt_qposadr[joint_id])
            low, high = model.actuator_ctrlrange[act_id]
            data.ctrl[act_id] = np.clip(data.qpos[qadr], low, high)

    mujoco.mj_forward(model, data)


def set_cube_pose(model, data, x, y, z=1.05):
    cube_id = get_body_id(model, CUBE_BODY)
    joint_adr = int(model.body_jntadr[cube_id])
    if joint_adr < 0:
        raise RuntimeError("orange_cube 没有 freejoint，无法设置位置")

    qadr = int(model.jnt_qposadr[joint_adr])

    data.qpos[qadr:qadr + 3] = np.array([x, y, z], dtype=np.float64)
    data.qpos[qadr + 3:qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    # 清一下 cube 速度
    dadr = int(model.jnt_dofadr[joint_adr])
    data.qvel[dadr:dadr + 6] = 0.0

    mujoco.mj_forward(model, data)


def body_pos(model, data, body_name):
    body_id = get_body_id(model, body_name)
    return data.xpos[body_id].copy()


def site_pos(model, data, site_name):
    site_id = get_site_id(model, site_name)
    return data.site_xpos[site_id].copy()


def is_success(model, data):
    cube = body_pos(model, data, CUBE_BODY)
    frame = body_pos(model, data, FRAME_BODY)

    xy_close = np.linalg.norm(cube[:2] - frame[:2]) < 0.045
    z_ok = cube[2] > frame[2] + 0.015

    return bool(xy_close and z_ok)


def is_lifted(cube_initial_z, model, data):
    cube = body_pos(model, data, CUBE_BODY)
    return bool(cube[2] > cube_initial_z + 0.015)


def solve_right_arm_ik(model, data, target_pos):
    site_id = get_site_id(model, RIGHT_SITE_NAME)

    joint_ids = [get_joint_id(model, name) for name in RIGHT_JOINT_NAMES]
    qadrs = [int(model.jnt_qposadr[jid]) for jid in joint_ids]
    dadrs = [int(model.jnt_dofadr[jid]) for jid in joint_ids]

    q_start = np.array([data.qpos[qadr] for qadr in qadrs], dtype=np.float64)

    for _ in range(IK_ITERS):
        mujoco.mj_forward(model, data)

        current = data.site_xpos[site_id].copy()
        err = target_pos - current

        if np.linalg.norm(err) < IK_TOL:
            q_sol = np.array([data.qpos[qadr] for qadr in qadrs], dtype=np.float64)

            # 还原
            for qadr, qv in zip(qadrs, q_start):
                data.qpos[qadr] = qv
            mujoco.mj_forward(model, data)

            return True, q_sol

        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

        J = jacp[:, dadrs]

        A = J @ J.T + IK_DAMPING * np.eye(3)
        dq = J.T @ np.linalg.solve(A, err)

        dq = np.clip(dq, -0.08, 0.08)

        for i, jid in enumerate(joint_ids):
            qadr = qadrs[i]
            q_new = data.qpos[qadr] + dq[i]

            if model.jnt_limited[jid]:
                low, high = model.jnt_range[jid]
                q_new = np.clip(q_new, low, high)

            data.qpos[qadr] = q_new

    q_sol = np.array([data.qpos[qadr] for qadr in qadrs], dtype=np.float64)

    # 还原
    for qadr, qv in zip(qadrs, q_start):
        data.qpos[qadr] = qv
    mujoco.mj_forward(model, data)

    return False, q_sol


def move_right_joints_to(model, data, q_target, steps=MOVE_STEPS):
    act_ids = [get_act_id(model, name) for name in RIGHT_ACTUATOR_NAMES]

    q_current = np.array([data.ctrl[act_id] for act_id in act_ids], dtype=np.float64)
    q_target = np.asarray(q_target, dtype=np.float64)

    for i in range(steps):
        alpha = (i + 1) / steps
        q = (1.0 - alpha) * q_current + alpha * q_target

        for act_id, qv in zip(act_ids, q):
            low, high = model.actuator_ctrlrange[act_id]
            data.ctrl[act_id] = np.clip(qv, low, high)

        mujoco.mj_step(model, data)


def move_right_tcp_to(model, data, target_pos, steps=MOVE_STEPS):
    ok, q_sol = solve_right_arm_ik(model, data, target_pos)
    if not ok:
        return False

    move_right_joints_to(model, data, q_sol, steps=steps)
    return True


def set_right_gripper(model, data, value, steps=OPEN_STEPS):
    act_id = get_act_id(model, RIGHT_FINGER_ACT)
    start = float(data.ctrl[act_id])
    target = float(value)

    low, high = model.actuator_ctrlrange[act_id]
    target = float(np.clip(target, low, high))

    for i in range(steps):
        alpha = (i + 1) / steps
        data.ctrl[act_id] = (1.0 - alpha) * start + alpha * target
        mujoco.mj_step(model, data)


def set_lifter(model, data, value, steps=LIFT_STEPS):
    act_id = get_act_id(model, LIFTER_ACT)
    start = float(data.ctrl[act_id])
    target = float(value)

    low, high = model.actuator_ctrlrange[act_id]
    target = float(np.clip(target, low, high))

    for i in range(steps):
        alpha = (i + 1) / steps
        data.ctrl[act_id] = (1.0 - alpha) * start + alpha * target
        mujoco.mj_step(model, data)


def run_one_position(model, x, y):
    data = mujoco.MjData(model)

    load_home(model, data)

    set_lifter(model, data, LIFTER_HOME, steps=20)
    set_right_gripper(model, data, RIGHT_FINGER_PRE_OPEN, steps=20)

    set_cube_pose(model, data, float(x), float(y), z=1.05)

    step_n(model, data, SETTLE_STEPS)

    cube_initial = body_pos(model, data, CUBE_BODY).copy()
    cube_initial_z = float(cube_initial[2])
    frame = body_pos(model, data, FRAME_BODY).copy()

    # 1. 张开右夹爪
    set_right_gripper(model, data, RIGHT_FINGER_PRE_OPEN, steps=OPEN_STEPS)

    # 2. 到 pregrasp
    pick_base = body_pos(model, data, CUBE_BODY).copy()
    pregrasp_target = pick_base + PICK_PREGRASP_OFFSET
    ok = move_right_tcp_to(model, data, pregrasp_target, steps=MOVE_STEPS)
    if not ok:
        return False, "ik_pregrasp_fail", body_pos(model, data, CUBE_BODY)

    # 3. 到 grasp
    grasp_target = pick_base + PICK_GRASP_OFFSET
    ok = move_right_tcp_to(model, data, grasp_target, steps=GRASP_MOVE_STEPS)
    if not ok:
        return False, "ik_grasp_fail", body_pos(model, data, CUBE_BODY)

    # 4. 闭合
    set_right_gripper(model, data, RIGHT_FINGER_CLOSE, steps=CLOSE_STEPS)
    step_n(model, data, HOLD_STEPS // 2)

    # 5. 抬起
    set_lifter(model, data, LIFTER_UP, steps=LIFT_STEPS)
    step_n(model, data, HOLD_STEPS)

    if not is_lifted(cube_initial_z, model, data):
        return False, "lift_fail", body_pos(model, data, CUBE_BODY)

    # 6. 到放置上方
    frame = body_pos(model, data, FRAME_BODY).copy()
    preplace_target = frame + PLACE_PREPLACE_OFFSET
    ok = move_right_tcp_to(model, data, preplace_target, steps=PLACE_MOVE_STEPS)
    if not ok:
        return False, "ik_preplace_fail", body_pos(model, data, CUBE_BODY)

    # 7. 到释放点
    frame = body_pos(model, data, FRAME_BODY).copy()
    release_target = frame + PLACE_RELEASE_OFFSET
    ok = move_right_tcp_to(model, data, release_target, steps=PLACE_MOVE_STEPS)
    if not ok:
        return False, "ik_release_fail", body_pos(model, data, CUBE_BODY)

    # 8. 张开释放
    set_right_gripper(model, data, RIGHT_FINGER_OPEN, steps=OPEN_STEPS)
    step_n(model, data, HOLD_STEPS)

    success = is_success(model, data)
    reason = "success" if success else "place_fail"

    return success, reason, body_pos(model, data, CUBE_BODY)


def main():
    print("加载模型:", XML_PATH)

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, RIGHT_SITE_NAME)
    if site_id < 0:
        raise RuntimeError(
            f"找不到 site: {RIGHT_SITE_NAME}\n"
            f"请先在 v2\\openarm_bimanual.xml 的 right_ee_control_point 下面添加：\n"
            f'<site name="right_gripper_tcp" pos="-0.00143 0 -0.105" size="0.012" rgba="0 0 1 1" />'
        )

    print("开始扫描右臂工作空间")
    print("X_VALUES:", X_VALUES)
    print("Y_VALUES:", Y_VALUES)
    print("")
    print("图例：")
    print("  O = 右臂 pick-and-place 成功")
    print("  X = 右臂失败")
    print("")

    results = {}
    reasons = {}

    for y in reversed(Y_VALUES):
        for x in X_VALUES:
            success, reason, cube_pos = run_one_position(model, x, y)

            results[(float(x), float(y))] = success
            reasons[(float(x), float(y))] = reason

            mark = "O" if success else "X"

            print(
                f"x={x:.3f}, y={y:.3f} -> {mark}, "
                f"reason={reason:18s}, "
                f"cube={np.array2string(cube_pos, precision=3)}"
            )

        print("")

    print("")
    print("=" * 70)
    print("右臂工作空间 ASCII 图")
    print("=" * 70)
    print("列是 x：", " ".join([f"{x:.3f}" for x in X_VALUES]))
    print("行是 y，从 +y 到 -y")
    print("")

    for y in reversed(Y_VALUES):
        row = []
        for x in X_VALUES:
            row.append("O" if results[(float(x), float(y))] else "X")

        print(f"y={y:+.3f}:  " + "   ".join(row))

    print("")
    print("=" * 70)
    print("失败原因统计")
    print("=" * 70)

    all_reasons = list(reasons.values())
    for r in sorted(set(all_reasons)):
        print(f"{r}: {all_reasons.count(r)}")

    print("")
    print("接下来对比左臂图：")
    print("如果右臂在左臂 X 的地方是 O，就值得训练右臂。")
    print("如果右臂也大面积 X，那要先改抓取/放置 expert，而不是直接训练。")


if __name__ == "__main__":
    main()
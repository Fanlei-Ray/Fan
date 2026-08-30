from pathlib import Path

import numpy as np
import mujoco


ROOT = Path(__file__).resolve().parents[1]
XML_PATH = ROOT / "v2" / "demo.xml"

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

LEFT_FINGER_ACT = "left_finger1_ctrl"
LIFTER_ACT = "lifter_ctrl"

CUBE_BODY = "orange_cube"
FRAME_BODY = "black_frame"

LEFT_FINGER_OPEN = 0.49
LEFT_FINGER_PRE_OPEN = 0.445
LEFT_FINGER_CLOSE = 0.0

LIFTER_HOME = 0.0
LIFTER_UP = 0.11

TEST_POINTS = [
    # 重点失败带
    (0.497, 0.050),
    (0.516, 0.050),
    (0.534, 0.050),
    (0.497, 0.075),
    (0.516, 0.075),
    (0.534, 0.075),

    # 对照点：v4 本来比较稳的位置
    (0.516, 0.000),
    (0.516, -0.050),
]

POSE_IK_ITERS = 220
POSE_IK_POS_TOL = 0.008
POSE_IK_ORI_TOL = 0.15
POSE_IK_DAMPING = 5e-3

POS_WEIGHT = 1.0
ORI_WEIGHT = 0.30
DQ_CLIP = 0.06

SETTLE_STEPS = 80
MOVE_STEPS = 120
GRASP_MOVE_STEPS = 140
CLOSE_STEPS = 120
LIFT_STEPS = 120
HOLD_STEPS = 120


OFFSET_MODES = [
    {
        "name": "normal",
        "pregrasp": np.array([-0.005, 0.000, 0.100], dtype=np.float64),
        "grasp": np.array([-0.010, 0.000, -0.005], dtype=np.float64),
    },
    {
        "name": "higher",
        "pregrasp": np.array([-0.005, 0.000, 0.110], dtype=np.float64),
        "grasp": np.array([-0.010, 0.000, 0.010], dtype=np.float64),
    },
    {
        "name": "x_plus",
        "pregrasp": np.array([0.005, 0.000, 0.100], dtype=np.float64),
        "grasp": np.array([0.010, 0.000, -0.005], dtype=np.float64),
    },
    {
        "name": "x_minus_more",
        "pregrasp": np.array([-0.015, 0.000, 0.100], dtype=np.float64),
        "grasp": np.array([-0.020, 0.000, -0.005], dtype=np.float64),
    },
    {
        "name": "y_plus",
        "pregrasp": np.array([-0.005, 0.012, 0.100], dtype=np.float64),
        "grasp": np.array([-0.010, 0.012, -0.005], dtype=np.float64),
    },
    {
        "name": "y_minus",
        "pregrasp": np.array([-0.005, -0.012, 0.100], dtype=np.float64),
        "grasp": np.array([-0.010, -0.012, -0.005], dtype=np.float64),
    },
]


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


def body_pos(model, data, name):
    body_id = get_body_id(model, name)
    return data.xpos[body_id].copy()


def site_pos(model, data, name):
    site_id = get_site_id(model, name)
    return data.site_xpos[site_id].copy()


def step_n(model, data, n):
    for _ in range(n):
        mujoco.mj_step(model, data)


def load_home(model, data):
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)

    mujoco.mj_forward(model, data)

    # 同步左臂 position actuator ctrl 到当前 qpos
    for act_name in LEFT_ACTUATOR_NAMES + [LEFT_FINGER_ACT, LIFTER_ACT]:
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
    dadr = int(model.jnt_dofadr[joint_adr])

    data.qpos[qadr:qadr + 3] = np.array([x, y, z], dtype=np.float64)
    data.qpos[qadr + 3:qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    data.qvel[dadr:dadr + 6] = 0.0

    mujoco.mj_forward(model, data)


def set_left_gripper(model, data, value, steps=80):
    act_id = get_act_id(model, LEFT_FINGER_ACT)
    start = float(data.ctrl[act_id])

    low, high = model.actuator_ctrlrange[act_id]
    target = float(np.clip(value, low, high))

    for i in range(steps):
        alpha = (i + 1) / steps
        data.ctrl[act_id] = (1.0 - alpha) * start + alpha * target
        mujoco.mj_step(model, data)


def set_lifter(model, data, value, steps=100):
    act_id = get_act_id(model, LIFTER_ACT)
    start = float(data.ctrl[act_id])

    low, high = model.actuator_ctrlrange[act_id]
    target = float(np.clip(value, low, high))

    for i in range(steps):
        alpha = (i + 1) / steps
        data.ctrl[act_id] = (1.0 - alpha) * start + alpha * target
        mujoco.mj_step(model, data)


def is_lifted(cube_initial_z, model, data):
    cube = body_pos(model, data, CUBE_BODY)
    return bool(cube[2] > cube_initial_z + 0.015)


def rotz(theta):
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def wrap_deg(angle):
    while angle > 180.0:
        angle -= 360.0
    while angle <= -180.0:
        angle += 360.0
    return angle


def unique_angles(angles):
    out = []

    for a in angles:
        a = wrap_deg(float(a))

        exists = False
        for b in out:
            if abs(wrap_deg(a - b)) < 1e-6:
                exists = True
                break

        if not exists:
            out.append(a)

    return out


def candidate_yaws_from_geometry(cube_pos, frame_pos):
    """
    根据 cube 和 black_frame 的相对位置生成候选抓取朝向。
    不是写死某个角度，而是每个位置自动生成候选。
    """
    dx = float(frame_pos[0] - cube_pos[0])
    dy = float(frame_pos[1] - cube_pos[1])

    to_frame_deg = np.rad2deg(np.arctan2(dy, dx))

    raw = [
        to_frame_deg,
        to_frame_deg + 180.0,
        to_frame_deg + 90.0,
        to_frame_deg - 90.0,

        # fallback 常用方向
        0.0,
        30.0,
        -30.0,
        45.0,
        -45.0,
        60.0,
        -60.0,
        90.0,
        -90.0,
        120.0,
        -120.0,
        150.0,
        -150.0,
        180.0,
    ]

    return unique_angles(raw)


def orientation_error(current_mat, target_mat):
    current_mat = current_mat.reshape(3, 3)
    target_mat = target_mat.reshape(3, 3)

    err = 0.5 * (
        np.cross(current_mat[:, 0], target_mat[:, 0])
        + np.cross(current_mat[:, 1], target_mat[:, 1])
        + np.cross(current_mat[:, 2], target_mat[:, 2])
    )

    return err.astype(np.float64)


def solve_left_arm_pose_ik(model, data, target_pos, target_mat):
    site_id = get_site_id(model, LEFT_SITE_NAME)

    joint_ids = [get_joint_id(model, name) for name in LEFT_JOINT_NAMES]
    qadrs = [int(model.jnt_qposadr[jid]) for jid in joint_ids]
    dadrs = [int(model.jnt_dofadr[jid]) for jid in joint_ids]

    q_start = np.array([data.qpos[qadr] for qadr in qadrs], dtype=np.float64)

    best_q = q_start.copy()
    best_score = float("inf")
    best_pos_err = float("inf")
    best_ori_err = float("inf")

    for _ in range(POSE_IK_ITERS):
        mujoco.mj_forward(model, data)

        current_pos = data.site_xpos[site_id].copy()
        current_mat = data.site_xmat[site_id].reshape(3, 3).copy()

        pos_err = target_pos - current_pos
        ori_err = orientation_error(current_mat, target_mat)

        pos_norm = float(np.linalg.norm(pos_err))
        ori_norm = float(np.linalg.norm(ori_err))

        score = pos_norm + 0.05 * ori_norm

        if score < best_score:
            best_score = score
            best_pos_err = pos_norm
            best_ori_err = ori_norm
            best_q = np.array([data.qpos[qadr] for qadr in qadrs], dtype=np.float64)

        if pos_norm < POSE_IK_POS_TOL and ori_norm < POSE_IK_ORI_TOL:
            q_sol = np.array([data.qpos[qadr] for qadr in qadrs], dtype=np.float64)

            for qadr, qv in zip(qadrs, q_start):
                data.qpos[qadr] = qv
            mujoco.mj_forward(model, data)

            return True, q_sol, pos_norm, ori_norm

        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

        J_pos = jacp[:, dadrs]
        J_ori = jacr[:, dadrs]

        J = np.vstack(
            [
                POS_WEIGHT * J_pos,
                ORI_WEIGHT * J_ori,
            ]
        )

        err = np.concatenate(
            [
                POS_WEIGHT * pos_err,
                ORI_WEIGHT * ori_err,
            ]
        )

        A = J @ J.T + POSE_IK_DAMPING * np.eye(6)
        dq = J.T @ np.linalg.solve(A, err)

        dq = np.clip(dq, -DQ_CLIP, DQ_CLIP)

        for i, jid in enumerate(joint_ids):
            qadr = qadrs[i]
            q_new = data.qpos[qadr] + dq[i]

            if model.jnt_limited[jid]:
                low, high = model.jnt_range[jid]
                q_new = np.clip(q_new, low, high)

            data.qpos[qadr] = q_new

    for qadr, qv in zip(qadrs, q_start):
        data.qpos[qadr] = qv
    mujoco.mj_forward(model, data)

    return False, best_q, best_pos_err, best_ori_err


def move_left_tcp_to_pose(model, data, target_pos, target_mat, steps):
    ok, q_sol, pos_err, ori_err = solve_left_arm_pose_ik(
        model=model,
        data=data,
        target_pos=target_pos,
        target_mat=target_mat,
    )

    if not ok:
        return False, pos_err, ori_err

    act_ids = [get_act_id(model, name) for name in LEFT_ACTUATOR_NAMES]

    q_current = np.array([data.ctrl[act_id] for act_id in act_ids], dtype=np.float64)
    q_target = np.asarray(q_sol, dtype=np.float64)

    for i in range(steps):
        alpha = (i + 1) / steps
        q = (1.0 - alpha) * q_current + alpha * q_target

        for act_id, qv in zip(act_ids, q):
            low, high = model.actuator_ctrlrange[act_id]
            data.ctrl[act_id] = np.clip(qv, low, high)

        mujoco.mj_step(model, data)

    return True, pos_err, ori_err


def get_home_left_site_mat(model, data):
    site_id = get_site_id(model, LEFT_SITE_NAME)
    mujoco.mj_forward(model, data)
    return data.site_xmat[site_id].reshape(3, 3).copy()


def attempt_pick_lift(model, x, y, yaw_deg, offset_mode):
    data = mujoco.MjData(model)

    load_home(model, data)

    set_lifter(model, data, LIFTER_HOME, steps=30)
    set_left_gripper(model, data, LEFT_FINGER_PRE_OPEN, steps=50)

    home_mat = get_home_left_site_mat(model, data)
    target_mat = rotz(np.deg2rad(yaw_deg)) @ home_mat

    set_cube_pose(model, data, float(x), float(y), z=1.05)
    step_n(model, data, SETTLE_STEPS)

    cube = body_pos(model, data, CUBE_BODY).copy()
    cube_initial_z = float(cube[2])

    pregrasp_target = cube + offset_mode["pregrasp"]
    grasp_target = cube + offset_mode["grasp"]

    ok, pos_err, ori_err = move_left_tcp_to_pose(
        model,
        data,
        pregrasp_target,
        target_mat,
        steps=MOVE_STEPS,
    )

    if not ok:
        return False, f"pregrasp_ik_fail_p{pos_err:.3f}_o{ori_err:.3f}", body_pos(model, data, CUBE_BODY)

    ok, pos_err, ori_err = move_left_tcp_to_pose(
        model,
        data,
        grasp_target,
        target_mat,
        steps=GRASP_MOVE_STEPS,
    )

    if not ok:
        return False, f"grasp_ik_fail_p{pos_err:.3f}_o{ori_err:.3f}", body_pos(model, data, CUBE_BODY)

    set_left_gripper(model, data, LEFT_FINGER_CLOSE, steps=CLOSE_STEPS)
    step_n(model, data, HOLD_STEPS // 2)

    set_lifter(model, data, LIFTER_UP, steps=LIFT_STEPS)
    step_n(model, data, HOLD_STEPS)

    lifted = is_lifted(cube_initial_z, model, data)

    if lifted:
        return True, "lift_success", body_pos(model, data, CUBE_BODY)

    return False, "lift_fail", body_pos(model, data, CUBE_BODY)


def run_one_point(model, x, y):
    data = mujoco.MjData(model)

    load_home(model, data)
    set_cube_pose(model, data, float(x), float(y), z=1.05)
    step_n(model, data, SETTLE_STEPS)

    cube = body_pos(model, data, CUBE_BODY).copy()
    frame = body_pos(model, data, FRAME_BODY).copy()

    yaw_candidates = candidate_yaws_from_geometry(cube, frame)

    print("")
    print("-" * 80)
    print(f"测试点 x={x:.3f}, y={y:.3f}")
    print("cube:", np.array2string(cube, precision=3))
    print("frame:", np.array2string(frame, precision=3))
    print("auto yaw candidates:", [round(a, 1) for a in yaw_candidates])
    print("-" * 80)

    attempts = []

    for offset_mode in OFFSET_MODES:
        for yaw_deg in yaw_candidates:
            success, reason, cube_pos = attempt_pick_lift(
                model=model,
                x=x,
                y=y,
                yaw_deg=yaw_deg,
                offset_mode=offset_mode,
            )

            record = {
                "success": success,
                "reason": reason,
                "yaw_deg": yaw_deg,
                "offset": offset_mode["name"],
                "cube_pos": cube_pos,
            }

            attempts.append(record)

            mark = "O" if success else "X"

            print(
                f"{mark} | "
                f"yaw={yaw_deg:7.1f} | "
                f"offset={offset_mode['name']:12s} | "
                f"reason={reason:28s} | "
                f"cube={np.array2string(cube_pos, precision=3)}"
            )

            # 这个点找到一个能抓起的姿态就够了
            if success:
                return record, attempts

    return None, attempts


def main():
    print("加载模型:", XML_PATH)

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        LEFT_SITE_NAME,
    )

    if site_id < 0:
        raise RuntimeError(
            f"找不到 site: {LEFT_SITE_NAME}\n"
            "请确认 v2\\openarm_bimanual.xml 里已经添加 left_gripper_tcp。"
        )

    print("")
    print("=" * 80)
    print("左臂自动抓取朝向校准")
    print("=" * 80)
    print("目标：验证 y=+0.050 / y=+0.075 失败带能否靠自动姿态修复。")
    print("当前只测试 pick + lift，不测试 place。")
    print("=" * 80)

    summary = []

    for x, y in TEST_POINTS:
        best, attempts = run_one_point(model, x, y)

        if best is None:
            summary.append((x, y, False, None))
        else:
            summary.append((x, y, True, best))

    print("")
    print("=" * 80)
    print("左臂自动 yaw 校准总结")
    print("=" * 80)

    for x, y, success, best in summary:
        if not success:
            print(f"x={x:.3f}, y={y:.3f}: X no successful grasp")
        else:
            print(
                f"x={x:.3f}, y={y:.3f}: O "
                f"yaw={best['yaw_deg']:.1f}, "
                f"offset={best['offset']}, "
                f"reason={best['reason']}"
            )

    success_count = sum(1 for _, _, success, _ in summary if success)

    print("")
    print(f"成功点数: {success_count}/{len(summary)}")

    print("")
    print("判断：")
    print("1. 如果 y=+0.050 / y=+0.075 多数出现 O，说明左臂可以靠自动抓取姿态修复失败带。")
    print("2. 如果只有对照点 O，失败带仍然 X，说明问题可能是放置/黑框碰撞或需要更复杂避障。")
    print("3. 如果成功 yaw 有规律，下一步把自动 yaw 逻辑写进左臂 expert，做 v7 数据。")


if __name__ == "__main__":
    main()
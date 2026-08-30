import numpy as np

import scan_right_workspace as rs


TEST_POINTS = [
    (0.516, 0.050),
    (0.516, 0.075),
    (0.516, 0.000),
    (0.497, 0.050),
    (0.534, 0.075),
]


POSE_IK_ITERS = 220
POSE_IK_POS_TOL = 0.008
POSE_IK_ORI_TOL = 0.15
POSE_IK_DAMPING = 5e-3

POS_WEIGHT = 1.0
ORI_WEIGHT = 0.30
DQ_CLIP = 0.06


FINGER_MODES = [
    {
        "name": "negative_open",
        "open": -0.49,
        "pre_open": -0.445,
        "close": 0.0,
    },
    {
        "name": "reverse",
        "open": 0.0,
        "pre_open": 0.0,
        "close": -0.49,
    },
]


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
    根据方块和黑框的位置自动生成候选抓取朝向。

    关键方向：
        cube -> frame
        frame -> cube
        垂直方向
        home fallback 方向

    注意：这里不是最终写死角度，而是每个位置自动生成候选。
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
        45.0,
        -45.0,
        90.0,
        -90.0,
        135.0,
        -135.0,
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


def solve_right_arm_pose_ik(model, data, target_pos, target_mat):
    site_id = rs.get_site_id(model, rs.RIGHT_SITE_NAME)

    joint_ids = [rs.get_joint_id(model, name) for name in rs.RIGHT_JOINT_NAMES]
    qadrs = [int(model.jnt_qposadr[jid]) for jid in joint_ids]
    dadrs = [int(model.jnt_dofadr[jid]) for jid in joint_ids]

    q_start = np.array([data.qpos[qadr] for qadr in qadrs], dtype=np.float64)

    best_q = q_start.copy()
    best_score = float("inf")
    best_pos_err = float("inf")
    best_ori_err = float("inf")

    for _ in range(POSE_IK_ITERS):
        rs.mujoco.mj_forward(model, data)

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
            rs.mujoco.mj_forward(model, data)

            return True, q_sol, pos_norm, ori_norm

        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        rs.mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

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
    rs.mujoco.mj_forward(model, data)

    return False, best_q, best_pos_err, best_ori_err


def move_right_tcp_to_pose(model, data, target_pos, target_mat, steps):
    ok, q_sol, pos_err, ori_err = solve_right_arm_pose_ik(
        model=model,
        data=data,
        target_pos=target_pos,
        target_mat=target_mat,
    )

    if not ok:
        return False, pos_err, ori_err

    rs.move_right_joints_to(model, data, q_sol, steps=steps)

    return True, pos_err, ori_err


def get_home_right_site_mat(model, data):
    site_id = rs.get_site_id(model, rs.RIGHT_SITE_NAME)
    rs.mujoco.mj_forward(model, data)
    return data.site_xmat[site_id].reshape(3, 3).copy()


def set_right_gripper_value(model, data, value, steps):
    act_id = rs.get_act_id(model, rs.RIGHT_FINGER_ACT)
    start = float(data.ctrl[act_id])

    low, high = model.actuator_ctrlrange[act_id]
    target = float(np.clip(value, low, high))

    for i in range(steps):
        alpha = (i + 1) / steps
        data.ctrl[act_id] = (1.0 - alpha) * start + alpha * target
        rs.mujoco.mj_step(model, data)


def attempt_pick_lift(model, x, y, yaw_deg, finger_mode, offset_mode):
    data = rs.mujoco.MjData(model)

    rs.load_home(model, data)

    rs.set_lifter(model, data, rs.LIFTER_HOME, steps=20)

    set_right_gripper_value(
        model,
        data,
        finger_mode["pre_open"],
        steps=40,
    )

    home_mat = get_home_right_site_mat(model, data)
    target_mat = rotz(np.deg2rad(yaw_deg)) @ home_mat

    rs.set_cube_pose(model, data, float(x), float(y), z=1.05)
    rs.step_n(model, data, rs.SETTLE_STEPS)

    cube_initial = rs.body_pos(model, data, rs.CUBE_BODY).copy()
    cube_initial_z = float(cube_initial[2])

    cube = rs.body_pos(model, data, rs.CUBE_BODY).copy()

    pregrasp_target = cube + offset_mode["pregrasp"]
    grasp_target = cube + offset_mode["grasp"]

    ok, pos_err, ori_err = move_right_tcp_to_pose(
        model,
        data,
        pregrasp_target,
        target_mat,
        steps=120,
    )

    if not ok:
        return False, f"pregrasp_ik_fail_p{pos_err:.3f}_o{ori_err:.3f}", rs.body_pos(model, data, rs.CUBE_BODY)

    ok, pos_err, ori_err = move_right_tcp_to_pose(
        model,
        data,
        grasp_target,
        target_mat,
        steps=140,
    )

    if not ok:
        return False, f"grasp_ik_fail_p{pos_err:.3f}_o{ori_err:.3f}", rs.body_pos(model, data, rs.CUBE_BODY)

    set_right_gripper_value(
        model,
        data,
        finger_mode["close"],
        steps=120,
    )

    rs.step_n(model, data, 80)

    rs.set_lifter(model, data, rs.LIFTER_UP, steps=120)
    rs.step_n(model, data, 120)

    lifted = rs.is_lifted(cube_initial_z, model, data)

    if lifted:
        return True, "lift_success", rs.body_pos(model, data, rs.CUBE_BODY)

    return False, "lift_fail", rs.body_pos(model, data, rs.CUBE_BODY)


def run_one_point(model, x, y):
    # 先用一次模型读出方块和框位置，生成自动候选角度
    data = rs.mujoco.MjData(model)
    rs.load_home(model, data)
    rs.set_cube_pose(model, data, float(x), float(y), z=1.05)
    rs.step_n(model, data, rs.SETTLE_STEPS)

    cube = rs.body_pos(model, data, rs.CUBE_BODY).copy()
    frame = rs.body_pos(model, data, rs.FRAME_BODY).copy()

    yaw_candidates = candidate_yaws_from_geometry(cube, frame)

    print("")
    print("-" * 80)
    print(f"测试点 x={x:.3f}, y={y:.3f}")
    print("cube:", np.array2string(cube, precision=3))
    print("frame:", np.array2string(frame, precision=3))
    print("auto yaw candidates:", [round(a, 1) for a in yaw_candidates])
    print("-" * 80)

    attempts = []

    for finger_mode in FINGER_MODES:
        for offset_mode in OFFSET_MODES:
            for yaw_deg in yaw_candidates:
                success, reason, cube_pos = attempt_pick_lift(
                    model=model,
                    x=x,
                    y=y,
                    yaw_deg=yaw_deg,
                    finger_mode=finger_mode,
                    offset_mode=offset_mode,
                )

                attempts.append(
                    {
                        "success": success,
                        "reason": reason,
                        "yaw_deg": yaw_deg,
                        "finger": finger_mode["name"],
                        "offset": offset_mode["name"],
                        "cube_pos": cube_pos,
                    }
                )

                mark = "O" if success else "X"

                print(
                    f"{mark} | "
                    f"yaw={yaw_deg:7.1f} | "
                    f"finger={finger_mode['name']:14s} | "
                    f"offset={offset_mode['name']:8s} | "
                    f"reason={reason:28s} | "
                    f"cube={np.array2string(cube_pos, precision=3)}"
                )

                if success:
                    return attempts[-1], attempts

    return None, attempts


def main():
    print("加载模型:", rs.XML_PATH)
    model = rs.mujoco.MjModel.from_xml_path(str(rs.XML_PATH))

    site_id = rs.mujoco.mj_name2id(
        model,
        rs.mujoco.mjtObj.mjOBJ_SITE,
        rs.RIGHT_SITE_NAME,
    )

    if site_id < 0:
        raise RuntimeError(f"找不到 site: {rs.RIGHT_SITE_NAME}")

    print("")
    print("=" * 80)
    print("右臂自动抓取朝向校准")
    print("=" * 80)
    print("不是写死 yaw，而是根据 cube->frame 自动生成候选方向。")
    print("每个候选会测试 finger mode + offset mode + pose IK。")
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
    print("右臂自动 yaw 校准总结")
    print("=" * 80)

    for x, y, success, best in summary:
        if not success:
            print(f"x={x:.3f}, y={y:.3f}: X no successful grasp")
        else:
            print(
                f"x={x:.3f}, y={y:.3f}: O "
                f"yaw={best['yaw_deg']:.1f}, "
                f"finger={best['finger']}, "
                f"offset={best['offset']}, "
                f"reason={best['reason']}"
            )

    success_count = sum(1 for _, _, success, _ in summary if success)

    print("")
    print(f"成功点数: {success_count}/{len(summary)}")

    print("")
    print("判断：")
    print("1. 如果这里出现 O，说明右臂可以通过自动选择抓取姿态解决。")
    print("2. 如果还是全 X，下一步就要在 viewer 里手动示教右臂第一条成功轨迹。")
    print("3. 只有右臂 expert 能稳定 pick+lift 后，才值得采右臂 BC 数据。")


if __name__ == "__main__":
    main()
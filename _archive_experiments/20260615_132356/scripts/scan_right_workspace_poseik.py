import numpy as np

import scan_right_workspace as rs


# 姿态 IK 参数
POSE_IK_ITERS = 180
POSE_IK_POS_TOL = 0.004
POSE_IK_ORI_TOL = 0.08
POSE_IK_DAMPING = 5e-3

POS_WEIGHT = 1.0
ORI_WEIGHT = 0.35

DQ_CLIP = 0.06


def orientation_error(current_mat, target_mat):
    """
    current_mat / target_mat: 3x3 rotation matrix.

    返回一个近似角速度误差向量。
    目标是让 current_mat 旋到 target_mat。
    """
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

        J = np.vstack([
            POS_WEIGHT * J_pos,
            ORI_WEIGHT * J_ori,
        ])

        err = np.concatenate([
            POS_WEIGHT * pos_err,
            ORI_WEIGHT * ori_err,
        ])

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

def rotz(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

def run_one_position_poseik(model, x, y):
    data = rs.mujoco.MjData(model)

    rs.load_home(model, data)

    rs.set_lifter(model, data, rs.LIFTER_HOME, steps=20)
    rs.set_right_gripper(model, data, rs.RIGHT_FINGER_PRE_OPEN, steps=20)

    # 关键：锁定 home 时右手爪的姿态
    # 后面移动到 cube / frame 时，都尽量保持这个姿态。
    target_mat = get_home_right_site_mat(model, data)
    target_mat = rotz(np.deg2rad(90.0)) @ home_mat

    rs.set_cube_pose(model, data, float(x), float(y), z=1.05)
    rs.step_n(model, data, rs.SETTLE_STEPS)

    cube_initial = rs.body_pos(model, data, rs.CUBE_BODY).copy()
    cube_initial_z = float(cube_initial[2])

    # 1. 张开右夹爪
    rs.set_right_gripper(model, data, rs.RIGHT_FINGER_PRE_OPEN, steps=rs.OPEN_STEPS)

    # 2. 到 pregrasp
    pick_base = rs.body_pos(model, data, rs.CUBE_BODY).copy()
    pregrasp_target = pick_base + rs.PICK_PREGRASP_OFFSET

    ok, pos_err, ori_err = move_right_tcp_to_pose(
        model,
        data,
        pregrasp_target,
        target_mat,
        steps=rs.MOVE_STEPS,
    )

    if not ok:
        return False, f"poseik_pregrasp_fail_p{pos_err:.3f}_o{ori_err:.3f}", rs.body_pos(model, data, rs.CUBE_BODY)

    # 3. 到 grasp
    grasp_target = pick_base + rs.PICK_GRASP_OFFSET

    ok, pos_err, ori_err = move_right_tcp_to_pose(
        model,
        data,
        grasp_target,
        target_mat,
        steps=rs.GRASP_MOVE_STEPS,
    )

    if not ok:
        return False, f"poseik_grasp_fail_p{pos_err:.3f}_o{ori_err:.3f}", rs.body_pos(model, data, rs.CUBE_BODY)

    # 4. 闭合
    rs.set_right_gripper(model, data, rs.RIGHT_FINGER_CLOSE, steps=rs.CLOSE_STEPS)
    rs.step_n(model, data, rs.HOLD_STEPS // 2)

    # 5. 抬起
    rs.set_lifter(model, data, rs.LIFTER_UP, steps=rs.LIFT_STEPS)
    rs.step_n(model, data, rs.HOLD_STEPS)

    if not rs.is_lifted(cube_initial_z, model, data):
        return False, "lift_fail", rs.body_pos(model, data, rs.CUBE_BODY)

    # 6. 到放置上方
    frame = rs.body_pos(model, data, rs.FRAME_BODY).copy()
    preplace_target = frame + rs.PLACE_PREPLACE_OFFSET

    ok, pos_err, ori_err = move_right_tcp_to_pose(
        model,
        data,
        preplace_target,
        target_mat,
        steps=rs.PLACE_MOVE_STEPS,
    )

    if not ok:
        return False, f"poseik_preplace_fail_p{pos_err:.3f}_o{ori_err:.3f}", rs.body_pos(model, data, rs.CUBE_BODY)

    # 7. 到释放点
    frame = rs.body_pos(model, data, rs.FRAME_BODY).copy()
    release_target = frame + rs.PLACE_RELEASE_OFFSET

    ok, pos_err, ori_err = move_right_tcp_to_pose(
        model,
        data,
        release_target,
        target_mat,
        steps=rs.PLACE_MOVE_STEPS,
    )

    if not ok:
        return False, f"poseik_release_fail_p{pos_err:.3f}_o{ori_err:.3f}", rs.body_pos(model, data, rs.CUBE_BODY)

    # 8. 张开释放
    rs.set_right_gripper(model, data, rs.RIGHT_FINGER_OPEN, steps=rs.OPEN_STEPS)
    rs.step_n(model, data, rs.HOLD_STEPS)

    success = rs.is_success(model, data)
    reason = "success" if success else "place_fail"

    return success, reason, rs.body_pos(model, data, rs.CUBE_BODY)


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

    print("开始扫描右臂工作空间：Pose IK 版本")
    print("X_VALUES:", rs.X_VALUES)
    print("Y_VALUES:", rs.Y_VALUES)
    print("")
    print("图例：")
    print("  O = 右臂 pose IK pick-and-place 成功")
    print("  X = 右臂失败")
    print("")

    results = {}
    reasons = {}

    for y in reversed(rs.Y_VALUES):
        for x in rs.X_VALUES:
            success, reason, cube_pos = run_one_position_poseik(model, x, y)

            results[(float(x), float(y))] = success
            reasons[(float(x), float(y))] = reason

            mark = "O" if success else "X"

            print(
                f"x={x:.3f}, y={y:.3f} -> {mark}, "
                f"reason={reason:32s}, "
                f"cube={np.array2string(cube_pos, precision=3)}"
            )

        print("")

    print("")
    print("=" * 70)
    print("右臂 Pose IK 工作空间 ASCII 图")
    print("=" * 70)
    print("列是 x：", " ".join([f"{x:.3f}" for x in rs.X_VALUES]))
    print("行是 y，从 +y 到 -y")
    print("")

    for y in reversed(rs.Y_VALUES):
        row = []
        for x in rs.X_VALUES:
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
    print("判断：")
    print("1. 如果 Pose IK 明显出现 O，说明右臂需要姿态 IK。")
    print("2. 如果还是全 lift_fail，下一步要做右臂手动示教/可视化单点调试。")
    print("3. 右臂 expert 不通之前，不要训练右臂 BC。")


if __name__ == "__main__":
    main()

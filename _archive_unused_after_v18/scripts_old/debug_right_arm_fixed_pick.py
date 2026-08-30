from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# 右臂 fixed-pick 调试脚本
#
# 目标：
#   只验证右臂能否在固定点抓起 orange_cube。
#
# 不做：
#   不做放置
#   不训练
#   不跑 BC
#   不跑 RL
#
# 成功标准：
#   cube 被抬高 max_lift_delta > 0.025
# ============================================================


ROOT = Path(__file__).resolve().parents[1]
XML_PATH = ROOT / "v2" / "demo.xml"


RENDER = False
RUN_VIEWER_FOR_BEST = True

# 先从左臂失败较多、应该交给右臂的正 y 区域开始
FIXED_CUBE_POS = np.array([0.516, 0.050, 1.050], dtype=float)

# 方块稳定后，最终实际 z 会略变，所以 x/y 更重要
CUBE_SETTLE_STEPS = 600

LIFTER_HOME = 0.0
LIFTER_UP = 0.11

RIGHT_FINGER_OPEN = -0.49
RIGHT_FINGER_PRE_OPEN = -0.445
RIGHT_FINGER_CLOSE = 0.0

LIFT_SUCCESS_DELTA = 0.025


RIGHT_SITE_CANDIDATES = [
    "right_gripper_tcp",
    "right_ee_control_point",
]

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


# ------------------------------------------------------------
# offset 说明
#
# 如果 XML 里有 right_gripper_tcp：
#   offset 是 right_gripper_tcp 相对 cube center 的目标偏移。
#
# 如果没有 right_gripper_tcp，只能用 right_ee_control_point：
#   offset 会比较大，因为 right_ee_control_point 在腕部附近。
# ------------------------------------------------------------

TCP_GRASP_OFFSETS = [
    ("tcp_x_p005_zp030", np.array([+0.005, 0.000, +0.030])),
    ("tcp_x_p005_zp025", np.array([+0.005, 0.000, +0.025])),
    ("tcp_x_p005_zp020", np.array([+0.005, 0.000, +0.020])),

    ("tcp_center_zp030", np.array([0.000, 0.000, +0.030])),
    ("tcp_center_zp025", np.array([0.000, 0.000, +0.025])),
    ("tcp_center_zp020", np.array([0.000, 0.000, +0.020])),

    ("tcp_x_m005_zp030", np.array([-0.005, 0.000, +0.030])),
    ("tcp_x_m005_zp025", np.array([-0.005, 0.000, +0.025])),

    ("tcp_y_m005_zp030", np.array([0.000, -0.005, +0.030])),
    ("tcp_y_p005_zp030", np.array([0.000, +0.005, +0.030])),
]

EE_GRASP_OFFSETS = [
    ("ee_mirror_left", np.array([-0.075, -0.045, -0.010])),
    ("ee_y_m025", np.array([-0.075, -0.025, -0.010])),
    ("ee_y_000", np.array([-0.075, 0.000, -0.010])),
    ("ee_y_p025", np.array([-0.075, +0.025, -0.010])),
    ("ee_y_p045", np.array([-0.075, +0.045, -0.010])),

    ("ee_x_m065_y_m045", np.array([-0.065, -0.045, -0.010])),
    ("ee_x_m085_y_m045", np.array([-0.085, -0.045, -0.010])),

    ("ee_x_m075_y_m045_z000", np.array([-0.075, -0.045, 0.000])),
    ("ee_x_m075_y_m045_zp010", np.array([-0.075, -0.045, +0.010])),
    ("ee_x_m075_y_m045_zm020", np.array([-0.075, -0.045, -0.020])),
]


POSTURE_CONFIGS = [
    ("j5_m060", {"openarm_right_joint5": -0.60}),
    ("j5_m075", {"openarm_right_joint5": -0.75}),
    ("j5_m090", {"openarm_right_joint5": -0.90}),
    ("j5_m060_j7_p030", {"openarm_right_joint5": -0.60, "openarm_right_joint7": +0.30}),
    ("j5_m060_j7_p060", {"openarm_right_joint5": -0.60, "openarm_right_joint7": +0.60}),
]


# ============================================================
# 基础工具
# ============================================================

def get_id(model, obj_type, name):
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id == -1:
        raise ValueError(f"找不到对象：{name}")
    return obj_id


def maybe_id(model, obj_type, name):
    return mujoco.mj_name2id(model, obj_type, name)


def joint_id(model, name):
    return get_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def actuator_id(model, name):
    return get_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def body_id(model, name):
    return get_id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def site_id(model, name):
    return get_id(model, mujoco.mjtObj.mjOBJ_SITE, name)


def get_body_pos(model, data, name):
    return data.xpos[body_id(model, name)].copy()


def get_site_pos(model, data, name):
    return data.site_xpos[site_id(model, name)].copy()


def set_ctrl(model, data, actuator_name, value):
    aid = actuator_id(model, actuator_name)
    low, high = model.actuator_ctrlrange[aid]
    data.ctrl[aid] = np.clip(value, low, high)


def choose_right_site(model):
    for name in RIGHT_SITE_CANDIDATES:
        if maybe_id(model, mujoco.mjtObj.mjOBJ_SITE, name) != -1:
            return name

    raise RuntimeError(
        "找不到右臂 site。请确认 XML 里存在 right_gripper_tcp 或 right_ee_control_point。"
    )


def set_free_body_pos(model, data, body_name, pos):
    bid = body_id(model, body_name)

    if model.body_jntnum[bid] < 1:
        raise RuntimeError(f"{body_name} 没有 freejoint，不能直接设置位置。")

    jid = model.body_jntadr[bid]
    qadr = model.jnt_qposadr[jid]
    dadr = model.jnt_dofadr[jid]

    # freejoint qpos: xyz + quat
    data.qpos[qadr:qadr + 3] = np.asarray(pos, dtype=float)
    data.qpos[qadr + 3:qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])

    # freejoint qvel: 6D
    data.qvel[dadr:dadr + 6] = 0.0

    mujoco.mj_forward(model, data)


def load_home(model, data):
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


def sync_position_actuators_to_qpos(model, data):
    for aid in range(model.nu):
        jid = model.actuator_trnid[aid, 0]
        if jid < 0:
            continue

        qaddr = model.jnt_qposadr[jid]
        low, high = model.actuator_ctrlrange[aid]
        data.ctrl[aid] = np.clip(data.qpos[qaddr], low, high)


def sim_steps(model, data, viewer=None, steps=100, realtime=False):
    for _ in range(steps):
        if viewer is not None and not viewer.is_running():
            return False

        step_start = time.time()

        mujoco.mj_step(model, data)

        if viewer is not None:
            viewer.sync()

        if realtime and viewer is not None:
            sleep_time = model.opt.timestep - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    return True


def move_to(model, data, targets, steps=500, viewer=None, realtime=False):
    start_ctrl = data.ctrl.copy()
    goal_ctrl = data.ctrl.copy()

    for name, value in targets.items():
        aid = actuator_id(model, name)
        low, high = model.actuator_ctrlrange[aid]
        goal_ctrl[aid] = np.clip(value, low, high)

    for i in range(steps):
        if viewer is not None and not viewer.is_running():
            return False

        alpha = (i + 1) / steps
        alpha = 3.0 * alpha ** 2 - 2.0 * alpha ** 3

        data.ctrl[:] = (1.0 - alpha) * start_ctrl + alpha * goal_ctrl

        step_start = time.time()
        mujoco.mj_step(model, data)

        if viewer is not None:
            viewer.sync()

        if realtime and viewer is not None:
            sleep_time = model.opt.timestep - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    return True


def close_right_gripper_slowly(
    model,
    data,
    start_value=RIGHT_FINGER_PRE_OPEN,
    end_value=RIGHT_FINGER_CLOSE,
    steps=700,
    viewer=None,
    realtime=False,
):
    aid = actuator_id(model, "right_finger1_ctrl")
    low, high = model.actuator_ctrlrange[aid]

    start_value = float(np.clip(start_value, low, high))
    end_value = float(np.clip(end_value, low, high))

    start_ctrl = data.ctrl.copy()
    goal_ctrl = data.ctrl.copy()
    goal_ctrl[aid] = end_value

    # 先确保从预张开开始
    data.ctrl[aid] = start_value
    sim_steps(model, data, viewer=viewer, steps=80, realtime=realtime)

    for i in range(steps):
        if viewer is not None and not viewer.is_running():
            return False

        alpha = (i + 1) / steps
        alpha = 3.0 * alpha ** 2 - 2.0 * alpha ** 3

        data.ctrl[aid] = (1.0 - alpha) * start_value + alpha * end_value

        step_start = time.time()
        mujoco.mj_step(model, data)

        if viewer is not None:
            viewer.sync()

        if realtime and viewer is not None:
            sleep_time = model.opt.timestep - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    return True


# ============================================================
# 右臂 IK
# ============================================================

def get_right_joint_info(model):
    qpos_addrs = []
    dof_addrs = []
    joint_ranges = []

    for name in RIGHT_JOINT_NAMES:
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


def build_posture_target(model, source_data, posture_overrides):
    q_target = []

    for name in RIGHT_JOINT_NAMES:
        jid = joint_id(model, name)
        qaddr = model.jnt_qposadr[jid]
        q = float(source_data.qpos[qaddr])

        if name in posture_overrides:
            q = float(posture_overrides[name])

        low, high = model.jnt_range[jid]
        q = float(np.clip(q, low, high))
        q_target.append(q)

    return np.array(q_target, dtype=float)


def solve_right_arm_ik(
    model,
    source_data,
    target_pos,
    site_name,
    posture_overrides=None,
    max_iters=240,
    tolerance=1e-3,
    step_size=0.65,
    damping=2e-3,
    posture_weight=0.018,
):
    if posture_overrides is None:
        posture_overrides = {}

    ik_data = make_ik_data(model, source_data)

    sid = site_id(model, site_name)
    qpos_addrs, dof_addrs, joint_ranges = get_right_joint_info(model)

    target_pos = np.asarray(target_pos, dtype=float)
    q_posture = build_posture_target(model, source_data, posture_overrides)

    best_error = 1e9

    for _ in range(max_iters):
        mujoco.mj_forward(model, ik_data)

        current_pos = ik_data.site_xpos[sid].copy()
        error = target_pos - current_pos
        err_norm = float(np.linalg.norm(error))

        best_error = min(best_error, err_norm)

        if err_norm < tolerance:
            break

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, ik_data, jacp, jacr, sid)

        J = jacp[:, dof_addrs]

        A = J @ J.T + damping * np.eye(3)
        dq_task = J.T @ np.linalg.solve(A, error)

        q_current = np.array([ik_data.qpos[qaddr] for qaddr in qpos_addrs], dtype=float)
        dq_posture = posture_weight * (q_posture - q_current)

        dq = step_size * dq_task + dq_posture
        dq = np.clip(dq, -0.055, 0.055)

        for i, qaddr in enumerate(qpos_addrs):
            ik_data.qpos[qaddr] += dq[i]

            low, high = joint_ranges[i]
            ik_data.qpos[qaddr] = np.clip(ik_data.qpos[qaddr], low, high)

    mujoco.mj_forward(model, ik_data)

    final_pos = ik_data.site_xpos[sid].copy()
    final_error = float(np.linalg.norm(target_pos - final_pos))

    ctrl_targets = {}

    for act_name, joint_name in zip(RIGHT_ACTUATOR_NAMES, RIGHT_JOINT_NAMES):
        aid = actuator_id(model, act_name)
        jid = joint_id(model, joint_name)
        qaddr = model.jnt_qposadr[jid]

        low, high = model.actuator_ctrlrange[aid]
        ctrl_targets[act_name] = float(np.clip(ik_data.qpos[qaddr], low, high))

    success = final_error < 0.015

    return success, final_error, best_error, ctrl_targets


def ik_move_right_to(
    model,
    data,
    target_pos,
    site_name,
    posture_overrides,
    move_steps,
    viewer=None,
    realtime=False,
):
    ik_ok, final_error, best_error, ctrl_targets = solve_right_arm_ik(
        model=model,
        source_data=data,
        target_pos=target_pos,
        site_name=site_name,
        posture_overrides=posture_overrides,
    )

    move_to(
        model,
        data,
        ctrl_targets,
        steps=move_steps,
        viewer=viewer,
        realtime=realtime,
    )

    sim_steps(model, data, viewer=viewer, steps=120, realtime=realtime)

    return ik_ok, final_error, best_error


# ============================================================
# 单次 fixed pick
# ============================================================

def reset_scene(model, data, cube_pos):
    load_home(model, data)

    set_free_body_pos(model, data, "orange_cube", cube_pos)

    set_ctrl(model, data, "lifter_ctrl", LIFTER_HOME)
    set_ctrl(model, data, "right_finger1_ctrl", RIGHT_FINGER_PRE_OPEN)

    sync_position_actuators_to_qpos(model, data)
    set_ctrl(model, data, "lifter_ctrl", LIFTER_HOME)
    set_ctrl(model, data, "right_finger1_ctrl", RIGHT_FINGER_PRE_OPEN)

    mujoco.mj_forward(model, data)


def run_fixed_pick_trial(
    model,
    site_name,
    grasp_offset,
    posture_overrides,
    viewer=None,
    realtime=False,
    label="trial",
):
    data = mujoco.MjData(model)

    reset_scene(model, data, FIXED_CUBE_POS)

    sim_steps(model, data, viewer=viewer, steps=CUBE_SETTLE_STEPS, realtime=realtime)

    cube_pos = get_body_pos(model, data, "orange_cube")
    cube_initial_z = float(cube_pos[2])

    # 每次稳定后重新锁定真实 cube 位置
    cube_pos = get_body_pos(model, data, "orange_cube")

    pregrasp_offset = grasp_offset + np.array([0.0, 0.0, 0.105], dtype=float)

    pregrasp_target = cube_pos + pregrasp_offset
    grasp_target = cube_pos + grasp_offset

    # 1. 右夹爪预张开
    move_to(
        model,
        data,
        {"right_finger1_ctrl": RIGHT_FINGER_PRE_OPEN, "lifter_ctrl": LIFTER_HOME},
        steps=350,
        viewer=viewer,
        realtime=realtime,
    )

    # 2. 到 pregrasp
    pre_ok, pre_err, pre_best = ik_move_right_to(
        model=model,
        data=data,
        target_pos=pregrasp_target,
        site_name=site_name,
        posture_overrides=posture_overrides,
        move_steps=700,
        viewer=viewer,
        realtime=realtime,
    )

    # 3. 下探到 grasp
    grasp_ok, grasp_err, grasp_best = ik_move_right_to(
        model=model,
        data=data,
        target_pos=grasp_target,
        site_name=site_name,
        posture_overrides=posture_overrides,
        move_steps=550,
        viewer=viewer,
        realtime=realtime,
    )

    # 4. 闭合夹爪
    close_right_gripper_slowly(
        model,
        data,
        start_value=RIGHT_FINGER_PRE_OPEN,
        end_value=RIGHT_FINGER_CLOSE,
        steps=850,
        viewer=viewer,
        realtime=realtime,
    )

    sim_steps(model, data, viewer=viewer, steps=300, realtime=realtime)

    # 5. 抬 lifter
    move_to(
        model,
        data,
        {"lifter_ctrl": LIFTER_UP, "right_finger1_ctrl": RIGHT_FINGER_CLOSE},
        steps=650,
        viewer=viewer,
        realtime=realtime,
    )

    max_cube_z = cube_initial_z

    for _ in range(500):
        if viewer is not None and not viewer.is_running():
            break

        mujoco.mj_step(model, data)

        cube_now = get_body_pos(model, data, "orange_cube")
        max_cube_z = max(max_cube_z, float(cube_now[2]))

        if viewer is not None:
            viewer.sync()

        if realtime and viewer is not None:
            time.sleep(model.opt.timestep)

    final_cube = get_body_pos(model, data, "orange_cube")
    final_site = get_site_pos(model, data, site_name)

    max_lift_delta = float(max_cube_z - cube_initial_z)
    final_lift_delta = float(final_cube[2] - cube_initial_z)

    success = max_lift_delta > LIFT_SUCCESS_DELTA

    return {
        "label": label,
        "success": success,
        "cube_initial": cube_pos.copy(),
        "cube_final": final_cube.copy(),
        "site_final": final_site.copy(),
        "max_lift_delta": max_lift_delta,
        "final_lift_delta": final_lift_delta,
        "pre_ok": pre_ok,
        "pre_err": pre_err,
        "pre_best": pre_best,
        "grasp_ok": grasp_ok,
        "grasp_err": grasp_err,
        "grasp_best": grasp_best,
    }


# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 80)
    print("Right arm fixed-pick debug")
    print("=" * 80)
    print("XML:", XML_PATH)
    print("FIXED_CUBE_POS:", FIXED_CUBE_POS)
    print("RENDER:", RENDER)
    print("=" * 80)

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))

    right_site = choose_right_site(model)

    print("使用右臂 site:", right_site)

    if right_site == "right_gripper_tcp":
        offset_configs = TCP_GRASP_OFFSETS
        print("检测到 right_gripper_tcp，使用 TCP 小 offset 扫描。")
    else:
        offset_configs = EE_GRASP_OFFSETS
        print("没有检测到 right_gripper_tcp，使用 right_ee_control_point 大 offset 扫描。")
        print("后面最好还是在 XML 里加 right_gripper_tcp。")

    all_results = []

    for offset_name, grasp_offset in offset_configs:
        for posture_name, posture_overrides in POSTURE_CONFIGS:
            label = f"{offset_name}__{posture_name}"

            result = run_fixed_pick_trial(
                model=model,
                site_name=right_site,
                grasp_offset=grasp_offset,
                posture_overrides=posture_overrides,
                viewer=None,
                realtime=False,
                label=label,
            )

            result["offset_name"] = offset_name
            result["posture_name"] = posture_name
            result["grasp_offset"] = grasp_offset.copy()

            all_results.append(result)

            mark = "O" if result["success"] else "X"

            print(
                f"{mark} {label:32s} "
                f"offset={np.array2string(grasp_offset, precision=3)} "
                f"lift_max={result['max_lift_delta']:.4f} "
                f"lift_final={result['final_lift_delta']:.4f} "
                f"pre_err={result['pre_err']:.4f} "
                f"grasp_err={result['grasp_err']:.4f}"
            )

    all_results.sort(
        key=lambda r: (
            r["success"],
            r["max_lift_delta"],
            -r["grasp_err"],
        ),
        reverse=True,
    )

    print("")
    print("=" * 80)
    print("TOP right fixed-pick configs")
    print("=" * 80)

    for r in all_results[:10]:
        mark = "O" if r["success"] else "X"
        print(
            f"{mark} {r['label']:32s} "
            f"offset={np.array2string(r['grasp_offset'], precision=3)} "
            f"lift_max={r['max_lift_delta']:.4f} "
            f"lift_final={r['final_lift_delta']:.4f} "
            f"pre_err={r['pre_err']:.4f} "
            f"grasp_err={r['grasp_err']:.4f} "
            f"cube_final={np.array2string(r['cube_final'], precision=4)}"
        )

    best = all_results[0]

    print("")
    print("=" * 80)
    print("BEST")
    print("=" * 80)
    print("label:", best["label"])
    print("success:", best["success"])
    print("offset_name:", best["offset_name"])
    print("posture_name:", best["posture_name"])
    print("grasp_offset:", best["grasp_offset"])
    print("max_lift_delta:", best["max_lift_delta"])
    print("final_lift_delta:", best["final_lift_delta"])
    print("pre_err:", best["pre_err"])
    print("grasp_err:", best["grasp_err"])

    if RUN_VIEWER_FOR_BEST:
        print("")
        print("=" * 80)
        print("打开 viewer 回放 BEST config")
        print("=" * 80)
        print("3 秒后开始。")

        data = mujoco.MjData(model)

        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.sync()
            time.sleep(3)

            run_fixed_pick_trial(
                model=model,
                site_name=right_site,
                grasp_offset=best["grasp_offset"],
                posture_overrides=dict(POSTURE_CONFIGS[[p[0] for p in POSTURE_CONFIGS].index(best["posture_name"])][1]),
                viewer=viewer,
                realtime=True,
                label=best["label"],
            )

            print("回放结束。viewer 保持运行。")
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
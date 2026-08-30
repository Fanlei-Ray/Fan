from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# Right arm fixed-pick clean debug
#
# 重点：
# 1. 不修改正常 demo 的 lifter / table / XML 初始化逻辑。
# 2. viewer 回放和 trial 使用同一个 data，避免 viewer 看错状态。
# 3. 只调右臂 fixed pick。
# ============================================================


ROOT = Path(__file__).resolve().parents[1]
XML_PATH = ROOT / "v2" / "demo.xml"

FIXED_CUBE_POS = np.array([0.516, 0.050, 1.050], dtype=float)

CUBE_SETTLE_STEPS = 800
LIFT_DELTA = 0.11
LIFT_SUCCESS_DELTA = 0.025

RIGHT_FINGER_OPEN = -0.49
RIGHT_FINGER_PRE_OPEN = -0.445
RIGHT_FINGER_CLOSE = 0.0

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


# TCP 已经放到夹爪中间附近，所以 offset 不要太极端。
TCP_GRASP_OFFSETS = [
    ("tcp_x_p005_z000", np.array([+0.005, 0.000, 0.000])),
    ("tcp_x_p005_zp005", np.array([+0.005, 0.000, +0.005])),
    ("tcp_x_p005_zp010", np.array([+0.005, 0.000, +0.010])),

    ("tcp_center_z000", np.array([0.000, 0.000, 0.000])),
    ("tcp_center_zp005", np.array([0.000, 0.000, +0.005])),
    ("tcp_center_zp010", np.array([0.000, 0.000, +0.010])),

    ("tcp_x_m005_z000", np.array([-0.005, 0.000, 0.000])),
    ("tcp_x_m005_zp005", np.array([-0.005, 0.000, +0.005])),

    ("tcp_y_m005_z000", np.array([0.000, -0.005, 0.000])),
    ("tcp_y_p005_z000", np.array([0.000, +0.005, 0.000])),
]


# 只围绕之前成功过的 j5_m060 附近扫。
POSTURE_CONFIGS = [
    ("j5_m060", {"openarm_right_joint5": -0.60}),
    ("j5_m075", {"openarm_right_joint5": -0.75}),
    ("j5_m090", {"openarm_right_joint5": -0.90}),

    ("j5_m060_j7_p030", {"openarm_right_joint5": -0.60, "openarm_right_joint7": +0.30}),
    ("j5_m060_j7_m030", {"openarm_right_joint5": -0.60, "openarm_right_joint7": -0.30}),

    ("j5_m075_j7_p030", {"openarm_right_joint5": -0.75, "openarm_right_joint7": +0.30}),
    ("j5_m075_j7_m030", {"openarm_right_joint5": -0.75, "openarm_right_joint7": -0.30}),
]


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


def get_ctrl(model, data, actuator_name):
    aid = actuator_id(model, actuator_name)
    return float(data.ctrl[aid])


def sync_position_actuators_to_qpos(model, data):
    """
    把 position actuator 的 ctrl 同步到当前 qpos。
    不额外改 lifter / table / 机械臂高度。
    """
    for aid in range(model.nu):
        jid = model.actuator_trnid[aid, 0]
        if jid < 0:
            continue

        qaddr = model.jnt_qposadr[jid]
        low, high = model.actuator_ctrlrange[aid]
        data.ctrl[aid] = np.clip(data.qpos[qaddr], low, high)


def load_normal_home(model, data):
    """
    只做正常 XML 初始化：
    1. 如果有 home keyframe，就加载 home。
    2. 把 position actuator ctrl 同步到 qpos。

    注意：
    不强行改 lifter。
    不强行改工作台。
    不修补正常 demo。
    """
    key_id = maybe_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        print("已加载 XML home keyframe。")
    else:
        print("没有 home keyframe，使用默认 qpos。")

    sync_position_actuators_to_qpos(model, data)
    mujoco.mj_forward(model, data)


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


def sim_steps(model, data, steps=100, viewer=None, realtime=False):
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


def close_right_gripper_slowly(model, data, viewer=None, realtime=False):
    aid = actuator_id(model, "right_finger1_ctrl")
    low, high = model.actuator_ctrlrange[aid]

    start_value = float(np.clip(RIGHT_FINGER_PRE_OPEN, low, high))
    end_value = float(np.clip(RIGHT_FINGER_CLOSE, low, high))

    data.ctrl[aid] = start_value
    sim_steps(model, data, steps=100, viewer=viewer, realtime=realtime)

    for i in range(850):
        if viewer is not None and not viewer.is_running():
            return False

        alpha = (i + 1) / 850
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
    posture_overrides,
    max_iters=260,
    tolerance=1e-3,
    step_size=0.62,
    damping=2e-3,
    posture_weight=0.025,
):
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

        q_current = np.array(
            [ik_data.qpos[qaddr] for qaddr in qpos_addrs],
            dtype=float,
        )

        dq_posture = posture_weight * (q_posture - q_current)

        dq = step_size * dq_task + dq_posture
        dq = np.clip(dq, -0.050, 0.050)

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

    return final_error < 0.015, final_error, best_error, ctrl_targets


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

    sim_steps(model, data, steps=120, viewer=viewer, realtime=realtime)

    return ik_ok, final_error, best_error


def reset_scene_like_normal_demo(model, data, viewer=None, realtime=False):
    """
    正常初始化，不修补 lifter。
    只额外做两件事：
    1. 把 cube 放到右臂测试点
    2. 右夹爪预张开
    """
    load_normal_home(model, data)

    set_free_body_pos(model, data, "orange_cube", FIXED_CUBE_POS)

    set_ctrl(model, data, "right_finger1_ctrl", RIGHT_FINGER_PRE_OPEN)

    mujoco.mj_forward(model, data)

    if viewer is not None:
        viewer.sync()

    sim_steps(
        model,
        data,
        steps=CUBE_SETTLE_STEPS,
        viewer=viewer,
        realtime=realtime,
    )

    # 方块稳定后重新放回测试点，再短暂稳定。
    # 这样每个 trial 初始 cube 一致。
    set_free_body_pos(model, data, "orange_cube", FIXED_CUBE_POS)

    set_ctrl(model, data, "right_finger1_ctrl", RIGHT_FINGER_PRE_OPEN)

    mujoco.mj_forward(model, data)

    if viewer is not None:
        viewer.sync()

    sim_steps(
        model,
        data,
        steps=200,
        viewer=viewer,
        realtime=realtime,
    )


def run_trial(
    model,
    offset_name,
    grasp_offset,
    posture_name,
    posture_overrides,
    viewer=None,
    realtime=False,
    data=None,
):
    """
    关键修复：
    如果 viewer 已经绑定了一个 data，就必须把同一个 data 传进来。
    不能在 run_trial 里面偷偷新建另一个 data。
    """
    if data is None:
        data = mujoco.MjData(model)

    reset_scene_like_normal_demo(
        model,
        data,
        viewer=viewer,
        realtime=realtime,
    )

    if maybe_id(model, mujoco.mjtObj.mjOBJ_SITE, RIGHT_SITE_NAME) == -1:
        raise RuntimeError(
            f"找不到 {RIGHT_SITE_NAME}。请先在 openarm_bimanual.xml 里添加 right_gripper_tcp。"
        )

    cube_pos = get_body_pos(model, data, "orange_cube")
    cube_initial_z = float(cube_pos[2])

    lifter_start = get_ctrl(model, data, "lifter_ctrl")

    lifter_aid = actuator_id(model, "lifter_ctrl")
    lifter_low, lifter_high = model.actuator_ctrlrange[lifter_aid]
    lifter_lift_target = float(
        np.clip(lifter_start + LIFT_DELTA, lifter_low, lifter_high)
    )

    pregrasp_target = cube_pos + grasp_offset + np.array([0.0, 0.0, 0.105])
    grasp_target = cube_pos + grasp_offset

    move_to(
        model,
        data,
        {"right_finger1_ctrl": RIGHT_FINGER_PRE_OPEN},
        steps=300,
        viewer=viewer,
        realtime=realtime,
    )

    pre_ok, pre_err, pre_best = ik_move_right_to(
        model=model,
        data=data,
        target_pos=pregrasp_target,
        site_name=RIGHT_SITE_NAME,
        posture_overrides=posture_overrides,
        move_steps=700,
        viewer=viewer,
        realtime=realtime,
    )

    grasp_ok, grasp_err, grasp_best = ik_move_right_to(
        model=model,
        data=data,
        target_pos=grasp_target,
        site_name=RIGHT_SITE_NAME,
        posture_overrides=posture_overrides,
        move_steps=550,
        viewer=viewer,
        realtime=realtime,
    )

    close_right_gripper_slowly(
        model,
        data,
        viewer=viewer,
        realtime=realtime,
    )

    sim_steps(
        model,
        data,
        steps=300,
        viewer=viewer,
        realtime=realtime,
    )

    move_to(
        model,
        data,
        {
            "lifter_ctrl": lifter_lift_target,
            "right_finger1_ctrl": RIGHT_FINGER_CLOSE,
        },
        steps=650,
        viewer=viewer,
        realtime=realtime,
    )

    max_cube_z = cube_initial_z

    for _ in range(500):
        if viewer is not None and not viewer.is_running():
            break

        step_start = time.time()

        mujoco.mj_step(model, data)

        cube_now = get_body_pos(model, data, "orange_cube")
        max_cube_z = max(max_cube_z, float(cube_now[2]))

        if viewer is not None:
            viewer.sync()

        if realtime and viewer is not None:
            sleep_time = model.opt.timestep - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    final_cube = get_body_pos(model, data, "orange_cube")
    final_tcp = get_site_pos(model, data, RIGHT_SITE_NAME)

    max_lift_delta = float(max_cube_z - cube_initial_z)
    final_lift_delta = float(final_cube[2] - cube_initial_z)

    success = max_lift_delta > LIFT_SUCCESS_DELTA

    return {
        "label": f"{offset_name}__{posture_name}",
        "success": success,

        "offset_name": offset_name,
        "posture_name": posture_name,
        "posture_overrides": posture_overrides,
        "grasp_offset": grasp_offset.copy(),

        "cube_initial": cube_pos.copy(),
        "cube_final": final_cube.copy(),
        "tcp_final": final_tcp.copy(),

        "max_lift_delta": max_lift_delta,
        "final_lift_delta": final_lift_delta,

        "pre_ok": pre_ok,
        "pre_err": pre_err,
        "pre_best": pre_best,

        "grasp_ok": grasp_ok,
        "grasp_err": grasp_err,
        "grasp_best": grasp_best,

        "lifter_start": lifter_start,
        "lifter_lift_target": lifter_lift_target,
    }


def main():
    print("=" * 80)
    print("Right arm fixed-pick clean debug")
    print("=" * 80)
    print("XML:", XML_PATH)
    print("FIXED_CUBE_POS:", FIXED_CUBE_POS)
    print("RIGHT_SITE_NAME:", RIGHT_SITE_NAME)
    print("=" * 80)

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))

    if maybe_id(model, mujoco.mjtObj.mjOBJ_SITE, RIGHT_SITE_NAME) == -1:
        raise RuntimeError(
            f"找不到 {RIGHT_SITE_NAME}。请先确认 openarm_bimanual.xml 里有 right_gripper_tcp。"
        )

    results = []

    for offset_name, grasp_offset in TCP_GRASP_OFFSETS:
        for posture_name, posture_overrides in POSTURE_CONFIGS:
            result = run_trial(
                model=model,
                offset_name=offset_name,
                grasp_offset=grasp_offset,
                posture_name=posture_name,
                posture_overrides=posture_overrides,
                viewer=None,
                realtime=False,
                data=None,
            )

            results.append(result)

            mark = "O" if result["success"] else "X"

            print(
                f"{mark} {result['label']:36s} "
                f"offset={np.array2string(result['grasp_offset'], precision=3)} "
                f"lift_max={result['max_lift_delta']:.4f} "
                f"lift_final={result['final_lift_delta']:.4f} "
                f"pre_err={result['pre_err']:.4f} "
                f"grasp_err={result['grasp_err']:.4f} "
                f"lifter={result['lifter_start']:.3f}->{result['lifter_lift_target']:.3f}"
            )

    results.sort(
        key=lambda r: (
            r["success"],
            r["max_lift_delta"],
            -r["grasp_err"],
        ),
        reverse=True,
    )

    print("")
    print("=" * 80)
    print("TOP right fixed-pick clean configs")
    print("=" * 80)

    for r in results[:10]:
        mark = "O" if r["success"] else "X"
        print(
            f"{mark} {r['label']:36s} "
            f"offset={np.array2string(r['grasp_offset'], precision=3)} "
            f"lift_max={r['max_lift_delta']:.4f} "
            f"lift_final={r['final_lift_delta']:.4f} "
            f"pre_err={r['pre_err']:.4f} "
            f"grasp_err={r['grasp_err']:.4f} "
            f"cube_final={np.array2string(r['cube_final'], precision=4)}"
        )

    best = results[0]

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
    print("lifter:", best["lifter_start"], "->", best["lifter_lift_target"])

    print("")
    print("=" * 80)
    print("打开 viewer 回放 BEST")
    print("=" * 80)
    print("这只是 debug 回放。正常基准仍然是：openarm-mujoco-launch v2\\demo.xml")
    print("1 秒后开始。")

    data = mujoco.MjData(model)

    # 先把 viewer 绑定的这个 data 初始化成正常 demo + fixed cube 状态。
    # 注意：后面 run_trial 也会继续使用同一个 data，不会再偷偷新建另一个。
    reset_scene_like_normal_demo(
        model=model,
        data=data,
        viewer=None,
        realtime=False,
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        time.sleep(1)

        run_trial(
            model=model,
            offset_name=best["offset_name"],
            grasp_offset=best["grasp_offset"],
            posture_name=best["posture_name"],
            posture_overrides=best["posture_overrides"],
            viewer=viewer,
            realtime=True,
            data=data,
        )

        print("回放结束。viewer 保持运行。")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
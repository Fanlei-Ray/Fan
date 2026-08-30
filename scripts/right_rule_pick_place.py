from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# Right arm rule expert: fixed pick, auto search, then replay best
#
# 目的：
#   先把右臂规则 expert 跑通，再谈采集右臂 BC 数据。
#
# 正常基准仍然是：
#   openarm-mujoco-launch v2\demo.xml
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
XML_PATH = ROOT / "v2" / "demo.xml"

# 右臂先测试固定点。后面成功后再扩小范围随机。
FIXED_CUBE_POS = np.array([0.516, 0.050, 1.050], dtype=float)

# 如果 XML 里有 right_gripper_tcp，优先用它；没有就退回 wrist 点。
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

RIGHT_FINGER_OPEN = -0.49
RIGHT_FINGER_PRE_OPEN = -0.445
RIGHT_FINGER_CLOSE = 0.0

LIFTER_UP = 0.11
LIFT_SUCCESS_DELTA_Z = 0.015

# 运行模式
RUN_VIEWER_FOR_BEST = True
REPLAY_ONLY_IF_SUCCESS = False


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


def set_ctrl(model, data, name, value):
    aid = actuator_id(model, name)
    low, high = model.actuator_ctrlrange[aid]
    data.ctrl[aid] = np.clip(value, low, high)


def sync_position_actuators_to_qpos(model, data):
    for aid in range(model.nu):
        jid = model.actuator_trnid[aid, 0]
        if jid < 0:
            continue

        qaddr = model.jnt_qposadr[jid]
        low, high = model.actuator_ctrlrange[aid]
        data.ctrl[aid] = np.clip(data.qpos[qaddr], low, high)


def choose_right_site(model):
    for name in RIGHT_SITE_CANDIDATES:
        if maybe_id(model, mujoco.mjtObj.mjOBJ_SITE, name) != -1:
            return name

    raise RuntimeError(
        "找不到右臂控制 site。请确认 XML 中有 right_gripper_tcp 或 right_ee_control_point。"
    )


def load_home(model, data):
    key_id = maybe_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        print("已加载 home keyframe。")
    else:
        print("没有找到 home keyframe，使用默认姿态。")

    sync_position_actuators_to_qpos(model, data)
    mujoco.mj_forward(model, data)


def set_free_body_pos(model, data, body_name, pos):
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


def sim_steps(model, data, steps=100, viewer=None, realtime=False):
    for _ in range(steps):
        if viewer is not None and not viewer.is_running():
            return False

        step_start = time.time()
        mujoco.mj_step(model, data)

        if viewer is not None:
            viewer.sync()

        if realtime and viewer is not None and not getattr(viewer, "handles_realtime_pacing", False):
            sleep_time = model.opt.timestep - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    return True


def move_to(model, data, targets, duration=1.5, viewer=None, realtime=False):
    start_ctrl = data.ctrl.copy()
    goal_ctrl = data.ctrl.copy()

    for name, value in targets.items():
        aid = actuator_id(model, name)
        low, high = model.actuator_ctrlrange[aid]
        goal_ctrl[aid] = np.clip(value, low, high)

    dt = model.opt.timestep
    steps = max(1, int(duration / dt))

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

        if realtime and viewer is not None and not getattr(viewer, "handles_realtime_pacing", False):
            sleep_time = dt - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    return True


def close_right_gripper_slowly(model, data, duration=2.3, viewer=None, realtime=False):
    aid = actuator_id(model, "right_finger1_ctrl")
    low, high = model.actuator_ctrlrange[aid]

    start_value = float(np.clip(RIGHT_FINGER_PRE_OPEN, low, high))
    end_value = float(np.clip(RIGHT_FINGER_CLOSE, low, high))

    data.ctrl[aid] = start_value
    sim_steps(model, data, steps=100, viewer=viewer, realtime=realtime)

    dt = model.opt.timestep
    steps = max(1, int(duration / dt))

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

        if realtime and viewer is not None and not getattr(viewer, "handles_realtime_pacing", False):
            sleep_time = dt - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    return True


# ============================================================
# IK
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


def solve_right_arm_ik(
    model,
    source_data,
    target_pos,
    site_name,
    max_iters=260,
    tolerance=1e-3,
    step_size=0.62,
    damping=2e-3,
):
    ik_data = make_ik_data(model, source_data)

    sid = site_id(model, site_name)
    qpos_addrs, dof_addrs, joint_ranges = get_right_joint_info(model)

    target_pos = np.asarray(target_pos, dtype=float)
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
        dq = J.T @ np.linalg.solve(A, error)
        dq = step_size * dq
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


def apply_joint_biases(model, ctrl_targets, joint_biases):
    ctrl_targets = dict(ctrl_targets)

    for actuator_name, bias in joint_biases.items():
        aid = actuator_id(model, actuator_name)
        low, high = model.actuator_ctrlrange[aid]
        old_value = ctrl_targets.get(actuator_name, 0.0)
        ctrl_targets[actuator_name] = float(np.clip(old_value + bias, low, high))

    return ctrl_targets


def ik_move_right_to(
    model,
    data,
    target_pos,
    site_name,
    label,
    duration=1.5,
    joint_biases=None,
    viewer=None,
    realtime=False,
    settle_steps=160,
):
    if joint_biases is None:
        joint_biases = {}

    success, final_error, best_error, ctrl_targets = solve_right_arm_ik(
        model=model,
        source_data=data,
        target_pos=target_pos,
        site_name=site_name,
    )

    ctrl_targets = apply_joint_biases(model, ctrl_targets, joint_biases)

    move_to(
        model,
        data,
        ctrl_targets,
        duration=duration,
        viewer=viewer,
        realtime=realtime,
    )

    sim_steps(model, data, steps=settle_steps, viewer=viewer, realtime=realtime)

    actual_pos = get_site_pos(model, data, site_name)
    actual_error = float(np.linalg.norm(target_pos - actual_pos))

    return success, final_error, best_error, actual_error


# ============================================================
# 配置生成
# ============================================================

def make_tcp_configs():
    configs = []

    # 直接镜像左臂新版 gripper_tcp 成功结构：
    # 左臂 grasp offset 大约 [0, +0.02, +0.035]。
    # 右臂先扫 y 正负，避免镜像方向判断错。
    x_values = [-0.006, 0.000, +0.006]
    y_values = [-0.035, -0.025, -0.020, -0.015, -0.010, 0.000, +0.010, +0.020]
    z_values = [+0.025, +0.035, +0.045, +0.055]

    joint_bias_options = [
        ("bias_none", {}),
        ("j6_p060", {"right_joint6_ctrl": +0.060}),
        ("j6_m060", {"right_joint6_ctrl": -0.060}),
        ("j7_p060", {"right_joint7_ctrl": +0.060}),
        ("j7_m060", {"right_joint7_ctrl": -0.060}),
    ]

    for x in x_values:
        for y in y_values:
            for z in z_values:
                grasp_offset = np.array([x, y, z], dtype=float)
                pregrasp_offset = grasp_offset + np.array([0.0, 0.0, 0.085], dtype=float)

                for bias_name, joint_biases in joint_bias_options:
                    name = f"tcp_x{x:+.3f}_y{y:+.3f}_z{z:+.3f}__{bias_name}"
                    configs.append(
                        {
                            "name": name,
                            "site_type": "tcp",
                            "pregrasp_offset": pregrasp_offset,
                            "grasp_offset": grasp_offset,
                            "preplace_offset": np.array([0.0, 0.0, 0.14], dtype=float),
                            "place_offset": np.array([0.0, 0.0, 0.08], dtype=float),
                            "joint_biases": joint_biases,
                        }
                    )

    return configs


def make_wrist_configs():
    configs = []

    x_values = [-0.110, -0.085, -0.060, -0.035, 0.000, +0.035, +0.060, +0.085]
    y_values = [-0.080, -0.055, -0.030, 0.000, +0.030, +0.055, +0.080]
    z_values = [-0.020, -0.010, 0.000, +0.010, +0.020]

    joint_bias_options = [
        ("bias_none", {}),
        ("j6_p060", {"right_joint6_ctrl": +0.060}),
        ("j6_m060", {"right_joint6_ctrl": -0.060}),
    ]

    for x in x_values:
        for y in y_values:
            for z in z_values:
                grasp_offset = np.array([x, y, z], dtype=float)
                pregrasp_offset = grasp_offset + np.array([0.0, 0.0, 0.110], dtype=float)

                for bias_name, joint_biases in joint_bias_options:
                    name = f"wrist_x{x:+.3f}_y{y:+.3f}_z{z:+.3f}__{bias_name}"
                    configs.append(
                        {
                            "name": name,
                            "site_type": "wrist",
                            "pregrasp_offset": pregrasp_offset,
                            "grasp_offset": grasp_offset,
                            "preplace_offset": np.array([x, 0.0, 0.14], dtype=float),
                            "place_offset": np.array([x, 0.0, 0.08], dtype=float),
                            "joint_biases": joint_biases,
                        }
                    )

    return configs


# ============================================================
# 单次 trial
# ============================================================

def reset_scene(
    model,
    data,
    viewer=None,
    realtime=False,
    object_name="orange_cube",
    reset_home=True,
    reset_object=True,
    settle_scale=1.0,
):
    settle_scale = float(np.clip(settle_scale, 0.35, 2.0))

    def scaled_settle(steps, minimum):
        return max(int(minimum), int(round(float(steps) * settle_scale)))

    if reset_home:
        load_home(model, data)

    if reset_object:
        set_free_body_pos(model, data, object_name, FIXED_CUBE_POS)
    set_ctrl(model, data, "right_finger1_ctrl", RIGHT_FINGER_PRE_OPEN)

    mujoco.mj_forward(model, data)

    if viewer is not None:
        viewer.sync()

    sim_steps(
        model,
        data,
        steps=scaled_settle(700, 240),
        viewer=viewer,
        realtime=realtime,
    )

    # Calibration trials can request an exact repeatable spawn.  Live visual
    # execution disables this branch and preserves the perceived object pose.
    if reset_object:
        set_free_body_pos(model, data, object_name, FIXED_CUBE_POS)
    set_ctrl(model, data, "right_finger1_ctrl", RIGHT_FINGER_PRE_OPEN)
    mujoco.mj_forward(model, data)

    if viewer is not None:
        viewer.sync()

    sim_steps(
        model,
        data,
        steps=scaled_settle(180, 90),
        viewer=viewer,
        realtime=realtime,
    )


def run_trial(
    model,
    site_name,
    config,
    do_place=False,
    viewer=None,
    realtime=False,
    data=None,
    object_name="orange_cube",
    target_name="black_frame",
    reset_home=True,
    speed_scale=1.0,
    post_release_retreat=False,
    reset_object=True,
    perceived_object_pos=None,
):
    speed_scale = float(np.clip(speed_scale, 0.35, 2.0))

    def scaled_duration(seconds):
        return max(0.12, float(seconds) * speed_scale)

    def scaled_steps(steps, minimum=80):
        return max(int(minimum), int(round(float(steps) * speed_scale)))

    if data is None:
        data = mujoco.MjData(model)

    reset_scene(
        model,
        data,
        viewer=viewer,
        realtime=realtime,
        object_name=object_name,
        reset_home=reset_home,
        reset_object=reset_object,
        settle_scale=speed_scale if not reset_object else 1.0,
    )

    actual_cube_pos = get_body_pos(model, data, object_name)
    cube_pos = (
        actual_cube_pos.copy()
        if perceived_object_pos is None
        else np.asarray(perceived_object_pos, dtype=float).copy()
    )
    frame_pos = get_body_pos(model, data, target_name)
    cube_initial_z = float(actual_cube_pos[2])

    # 1. 右夹爪预张开
    move_to(
        model,
        data,
        {"right_finger1_ctrl": RIGHT_FINGER_PRE_OPEN},
        duration=scaled_duration(0.8),
        viewer=viewer,
        realtime=realtime,
    )

    # 2. pregrasp
    pregrasp_target = cube_pos + config["pregrasp_offset"]
    pre_ok, pre_err, pre_best, pre_actual_err = ik_move_right_to(
        model=model,
        data=data,
        target_pos=pregrasp_target,
        site_name=site_name,
        label="pregrasp",
        duration=scaled_duration(1.7),
        joint_biases=config["joint_biases"],
        viewer=viewer,
        realtime=realtime,
        settle_steps=scaled_steps(160),
    )

    # 3. grasp
    cube_pos = (
        get_body_pos(model, data, object_name)
        if perceived_object_pos is None
        else np.asarray(perceived_object_pos, dtype=float)
    )
    grasp_target = cube_pos + config["grasp_offset"]
    grasp_ok, grasp_err, grasp_best, grasp_actual_err = ik_move_right_to(
        model=model,
        data=data,
        target_pos=grasp_target,
        site_name=site_name,
        label="grasp",
        duration=scaled_duration(1.3),
        joint_biases=config["joint_biases"],
        viewer=viewer,
        realtime=realtime,
        settle_steps=scaled_steps(160),
    )

    # 4. 慢慢闭合右夹爪
    close_right_gripper_slowly(
        model,
        data,
        duration=scaled_duration(2.2),
        viewer=viewer,
        realtime=realtime,
    )
    sim_steps(
        model,
        data,
        steps=scaled_steps(300, minimum=160),
        viewer=viewer,
        realtime=realtime,
    )

    # 5. 抬 lifter
    move_to(
        model,
        data,
        {
            "lifter_ctrl": LIFTER_UP,
            "right_finger1_ctrl": RIGHT_FINGER_CLOSE,
        },
        duration=scaled_duration(1.5),
        viewer=viewer,
        realtime=realtime,
    )

    max_cube_z = cube_initial_z
    for _ in range(scaled_steps(500, minimum=260)):
        if viewer is not None and not viewer.is_running():
            break

        step_start = time.time()
        mujoco.mj_step(model, data)
        cube_now = get_body_pos(model, data, object_name)
        max_cube_z = max(max_cube_z, float(cube_now[2]))

        if viewer is not None:
            viewer.sync()

        if realtime and viewer is not None and not getattr(viewer, "handles_realtime_pacing", False):
            sleep_time = model.opt.timestep - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    cube_after_lift = get_body_pos(model, data, object_name)
    lift_delta = float(max_cube_z - cube_initial_z)
    final_lift_delta = float(cube_after_lift[2] - cube_initial_z)
    pick_success = lift_delta > LIFT_SUCCESS_DELTA_Z

    place_success = False
    final_cube = cube_after_lift.copy()

    # 6. 如果抓起成功，才尝试放置
    if do_place and pick_success:
        frame_pos = get_body_pos(model, data, target_name)

        preplace_target = frame_pos + config["preplace_offset"]
        ik_move_right_to(
            model=model,
            data=data,
            target_pos=preplace_target,
            site_name=site_name,
            label="preplace",
            duration=scaled_duration(1.8),
            joint_biases={},
            viewer=viewer,
            realtime=realtime,
            settle_steps=scaled_steps(160),
        )

        frame_pos = get_body_pos(model, data, target_name)
        place_target = frame_pos + config["place_offset"]
        ik_move_right_to(
            model=model,
            data=data,
            target_pos=place_target,
            site_name=site_name,
            label="place",
            duration=scaled_duration(1.4),
            joint_biases={},
            viewer=viewer,
            realtime=realtime,
            settle_steps=scaled_steps(160),
        )

        move_to(
            model,
            data,
            {"right_finger1_ctrl": RIGHT_FINGER_OPEN},
            duration=scaled_duration(1.0),
            viewer=viewer,
            realtime=realtime,
        )
        sim_steps(
            model,
            data,
            steps=scaled_steps(180, minimum=100),
            viewer=viewer,
            realtime=realtime,
        )

        # Tall or handled objects can remain lightly hooked on a finger after
        # opening.  A vertical retreat is the standard release sequence and is
        # optional so legacy experiments keep their exact historical motion.
        if post_release_retreat:
            frame_pos = get_body_pos(model, data, target_name)
            release_retreat_target = frame_pos + config["preplace_offset"]
            ik_move_right_to(
                model=model,
                data=data,
                target_pos=release_retreat_target,
                site_name=site_name,
                label="release_retreat",
                duration=scaled_duration(0.9),
                joint_biases={},
                viewer=viewer,
                realtime=realtime,
                settle_steps=scaled_steps(120),
            )

        sim_steps(
            model,
            data,
            steps=scaled_steps(320, minimum=180),
            viewer=viewer,
            realtime=realtime,
        )

        final_cube = get_body_pos(model, data, object_name)
        frame_pos = get_body_pos(model, data, target_name)
        xy_dist = float(np.linalg.norm(final_cube[:2] - frame_pos[:2]))
        z_margin = float(final_cube[2] - frame_pos[2])
        place_success = (xy_dist < 0.055) and (z_margin > 0.005)

    final_cube = get_body_pos(model, data, object_name)
    frame_pos = get_body_pos(model, data, target_name)
    xy_dist = float(np.linalg.norm(final_cube[:2] - frame_pos[:2]))
    z_margin = float(final_cube[2] - frame_pos[2])

    return {
        "config_name": config["name"],
        "site_type": config["site_type"],
        "pick_success": pick_success,
        "place_success": place_success,
        "lift_delta": lift_delta,
        "final_lift_delta": final_lift_delta,
        "pre_err": pre_err,
        "pre_actual_err": pre_actual_err,
        "grasp_err": grasp_err,
        "grasp_actual_err": grasp_actual_err,
        "xy_dist": xy_dist,
        "z_margin": z_margin,
        "speed_scale": speed_scale,
        "post_release_retreat": bool(post_release_retreat),
        "uses_perceived_object_pos": perceived_object_pos is not None,
        "cube_final": final_cube.copy(),
        "frame_final": frame_pos.copy(),
        "config": config,
    }


# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 80)
    print("Right arm rule expert: fixed pick/place auto search")
    print("=" * 80)
    print("XML:", XML_PATH)
    print("FIXED_CUBE_POS:", FIXED_CUBE_POS)
    print("=" * 80)

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    site_name = choose_right_site(model)

    print("使用右臂 site:", site_name)

    if site_name == "right_gripper_tcp":
        configs = make_tcp_configs()
        print("检测到 right_gripper_tcp：使用 gripper TCP 镜像左臂新版 offset 扫描。")
    else:
        configs = make_wrist_configs()
        print("没有 right_gripper_tcp：退回 right_ee_control_point 大 offset 扫描。")

    print("候选配置数量:", len(configs))
    print("开始无 viewer 快速扫描 fixed pick...")
    print("=" * 80)

    results = []

    for idx, config in enumerate(configs, start=1):
        result = run_trial(
            model=model,
            site_name=site_name,
            config=config,
            do_place=False,
            viewer=None,
            realtime=False,
            data=None,
        )
        results.append(result)

        mark = "O" if result["pick_success"] else "X"
        print(
            f"{mark} {idx:03d}/{len(configs):03d} "
            f"{config['name']:38s} "
            f"lift={result['lift_delta']:.4f} "
            f"final_lift={result['final_lift_delta']:.4f} "
            f"pre_err={result['pre_actual_err']:.4f} "
            f"grasp_err={result['grasp_actual_err']:.4f} "
            f"cube_final={np.array2string(result['cube_final'], precision=4)}"
        )

    results.sort(
        key=lambda r: (
            r["pick_success"],
            r["lift_delta"],
            r["final_lift_delta"],
            -r["grasp_actual_err"],
        ),
        reverse=True,
    )

    print("")
    print("=" * 80)
    print("TOP right rule expert configs")
    print("=" * 80)

    for r in results[:15]:
        mark = "O" if r["pick_success"] else "X"
        print(
            f"{mark} {r['config_name']:38s} "
            f"lift={r['lift_delta']:.4f} "
            f"final_lift={r['final_lift_delta']:.4f} "
            f"pre_err={r['pre_actual_err']:.4f} "
            f"grasp_err={r['grasp_actual_err']:.4f} "
            f"xy={r['xy_dist']:.4f} "
            f"z_margin={r['z_margin']:.4f} "
            f"cube_final={np.array2string(r['cube_final'], precision=4)}"
        )

    best = results[0]

    print("")
    print("=" * 80)
    print("BEST")
    print("=" * 80)
    print("config_name:", best["config_name"])
    print("site_type:", best["site_type"])
    print("pick_success:", best["pick_success"])
    print("lift_delta:", best["lift_delta"])
    print("final_lift_delta:", best["final_lift_delta"])
    print("pre_actual_err:", best["pre_actual_err"])
    print("grasp_actual_err:", best["grasp_actual_err"])
    print("pregrasp_offset:", best["config"]["pregrasp_offset"])
    print("grasp_offset:", best["config"]["grasp_offset"])
    print("joint_biases:", best["config"]["joint_biases"])

    if RUN_VIEWER_FOR_BEST:
        if REPLAY_ONLY_IF_SUCCESS and not best["pick_success"]:
            print("\n没有成功配置，跳过 viewer 回放。")
            return

        print("")
        print("=" * 80)
        print("打开 viewer 回放 BEST")
        print("=" * 80)
        print("如果 best 已经 pick_success=True，这次会继续尝试 place。")
        print("1 秒后开始。")

        data = mujoco.MjData(model)
        reset_scene(model, data, viewer=None, realtime=False)

        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.sync()
            time.sleep(1)

            replay = run_trial(
                model=model,
                site_name=site_name,
                config=best["config"],
                do_place=bool(best["pick_success"]),
                viewer=viewer,
                realtime=True,
                data=data,
            )

            print("")
            print("=" * 80)
            print("REPLAY RESULT")
            print("=" * 80)
            print("pick_success:", replay["pick_success"])
            print("place_success:", replay["place_success"])
            print("lift_delta:", replay["lift_delta"])
            print("final_lift_delta:", replay["final_lift_delta"])
            print("xy_dist:", replay["xy_dist"])
            print("z_margin:", replay["z_margin"])
            print("cube_final:", replay["cube_final"])
            print("frame_final:", replay["frame_final"])

            print("\n回放结束。viewer 保持运行。")
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()

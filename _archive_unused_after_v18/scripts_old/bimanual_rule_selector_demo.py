from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# 右臂 rule expert：刚刚已经调通
import right_rule_pick_place as right_rule

# 左臂 rule expert：你项目里这个名字虽然叫 right_pick_place.py，
# 但内容是左臂 pick/place 脚本
import right_pick_place as left_rule


XML_PATH = ROOT / "v2" / "demo.xml"


# ============================================================
# Demo 测试点
#
# cube_y < 0.02 走左臂
# cube_y >= 0.02 走右臂
# ============================================================

TEST_CASES = [
    {
        "name": "left_case",
        "cube_pos": np.array([0.516, -0.035, 1.050], dtype=float),
    },
    {
        "name": "right_case",
        "cube_pos": np.array([0.516, 0.050, 1.050], dtype=float),
    },
]

SELECT_RIGHT_IF_CUBE_Y_GE = 0.020


# ============================================================
# 兼容 left_rule 里不同版本的变量名
# ============================================================

def get_attr_any(module, names, default=None):
    for name in names:
        if hasattr(module, name):
            value = getattr(module, name)
            print(f"[LEFT CONFIG] {names[0]} <- left_rule.{name} = {value}")
            return value

    if default is not None:
        print(f"[LEFT CONFIG] {names[0]} 使用默认值 = {default}")
        return default

    raise AttributeError(f"left_rule 里找不到这些变量名：{names}")


LEFT_SITE_NAME = get_attr_any(
    left_rule,
    ["LEFT_SITE_NAME", "SITE_NAME", "LEFT_EE_SITE_NAME"],
    default="left_gripper_tcp",
)

LEFT_FINGER_OPEN = get_attr_any(
    left_rule,
    ["LEFT_FINGER_OPEN", "FINGER_OPEN"],
    default=0.49,
)

LEFT_FINGER_PRE_OPEN = get_attr_any(
    left_rule,
    ["LEFT_FINGER_PRE_OPEN", "FINGER_PRE_OPEN"],
    default=0.445,
)

LEFT_FINGER_CLOSE = get_attr_any(
    left_rule,
    ["LEFT_FINGER_CLOSE", "FINGER_CLOSE"],
    default=0.0,
)

# 先读取旧变量，方便打印确认，但下面会强制 override
LEFT_CUBE_PREGRASP_OFFSET = get_attr_any(
    left_rule,
    ["CUBE_PREGRASP_OFFSET", "PICK_PREGRASP_OFFSET"],
    default=np.array([-0.005, 0.000, 0.100], dtype=float),
)

LEFT_CUBE_GRASP_OFFSET = get_attr_any(
    left_rule,
    ["CUBE_GRASP_OFFSET", "PICK_GRASP_OFFSET"],
    default=np.array([-0.010, 0.000, -0.005], dtype=float),
)

LEFT_FRAME_PREPLACE_OFFSET = get_attr_any(
    left_rule,
    [
        "FRAME_PREPLACE_OFFSET",
        "PLACE_PREGRASP_OFFSET",
        "PLACE_PREPLACE_OFFSET",
        "PLACE_PRE_OFFSET",
    ],
    default=np.array([0.000, 0.000, 0.140], dtype=float),
)

LEFT_FRAME_PLACE_OFFSET = get_attr_any(
    left_rule,
    [
        "FRAME_PLACE_OFFSET",
        "PLACE_GRASP_OFFSET",
        "PLACE_OFFSET",
    ],
    default=np.array([0.000, 0.000, 0.080], dtype=float),
)

LIFTER_UP = get_attr_any(
    left_rule,
    ["LIFTER_UP"],
    default=0.11,
)

LIFT_SUCCESS_DELTA_Z = get_attr_any(
    left_rule,
    ["LIFT_SUCCESS_DELTA_Z"],
    default=0.015,
)


# ============================================================
# 强制修正左臂 gripper TCP offset
#
# 不修改 XML，不影响之前训练好的左臂 BC。
# 这里只让 bimanual rule selector demo 使用新的左臂 TCP 抓取偏移。
#
# 右臂成功配置：
#   pregrasp_offset = [-0.006, -0.020, 0.140]
#   grasp_offset    = [-0.006, -0.020, 0.055]
#
# 左臂这里做 y 方向镜像：
#   pregrasp_offset = [-0.006, +0.020, 0.140]
#   grasp_offset    = [-0.006, +0.020, 0.055]
# ============================================================

LEFT_CUBE_PREGRASP_OFFSET = np.array([-0.006, 0.020, 0.140], dtype=float)
LEFT_CUBE_GRASP_OFFSET = np.array([-0.006, 0.020, 0.055], dtype=float)

LEFT_FRAME_PREPLACE_OFFSET = np.array([0.000, 0.000, 0.140], dtype=float)
LEFT_FRAME_PLACE_OFFSET = np.array([0.000, 0.000, 0.080], dtype=float)

print("[LEFT OVERRIDE] 不修改 XML，只修改 selector demo 里的左臂 TCP offsets")
print("[LEFT OVERRIDE] LEFT_CUBE_PREGRASP_OFFSET:", LEFT_CUBE_PREGRASP_OFFSET)
print("[LEFT OVERRIDE] LEFT_CUBE_GRASP_OFFSET:", LEFT_CUBE_GRASP_OFFSET)
print("[LEFT OVERRIDE] LEFT_FRAME_PREPLACE_OFFSET:", LEFT_FRAME_PREPLACE_OFFSET)
print("[LEFT OVERRIDE] LEFT_FRAME_PLACE_OFFSET:", LEFT_FRAME_PLACE_OFFSET)

LEFT_PICK_JOINT_BIASES = {
       "left_joint7_ctrl": 0.060,
}

print("[LEFT OVERRIDE] LEFT_PICK_JOINT_BIASES:", LEFT_PICK_JOINT_BIASES) 

# ============================================================
# 右臂 best config，来自右臂 fixed pick/place 成功结果
# ============================================================

RIGHT_BEST_CONFIG = {
    "name": "right_best_tcp_x-0.006_y-0.020_z+0.055__j7_m060",
    "site_type": "tcp",

    "pregrasp_offset": np.array([-0.006, -0.020, 0.140], dtype=float),
    "grasp_offset": np.array([-0.006, -0.020, 0.055], dtype=float),

    "preplace_offset": np.array([0.0, 0.0, 0.140], dtype=float),
    "place_offset": np.array([0.0, 0.0, 0.080], dtype=float),

    "joint_biases": {
        "right_joint7_ctrl": -0.060,
    },
}


# ============================================================
# 基础工具
# ============================================================

def maybe_id(model, obj_type, name):
    return mujoco.mj_name2id(model, obj_type, name)


def get_id(model, obj_type, name):
    obj_id = mujoco.mj_name2id(model, obj_type, name)

    if obj_id == -1:
        raise ValueError(f"找不到对象：{name}")

    return obj_id


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
    data.ctrl[aid] = np.clip(float(value), low, high)


def sync_position_actuators_to_qpos(model, data):
    for aid in range(model.nu):
        jid = model.actuator_trnid[aid, 0]

        if jid < 0:
            continue

        qaddr = model.jnt_qposadr[jid]
        low, high = model.actuator_ctrlrange[aid]
        data.ctrl[aid] = np.clip(data.qpos[qaddr], low, high)


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


def load_home_quiet(model, data):
    key_id = maybe_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        print("已加载 home keyframe。")
    else:
        print("没有找到 home keyframe，使用默认姿态。")

    sync_position_actuators_to_qpos(model, data)
    mujoco.mj_forward(model, data)


def sim_steps(model, data, steps, viewer=None, realtime=False):
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


def move_to_ctrl(model, data, targets, duration=1.5, viewer=None, realtime=False):
    start_ctrl = data.ctrl.copy()
    goal_ctrl = data.ctrl.copy()

    for name, value in targets.items():
        aid = actuator_id(model, name)
        low, high = model.actuator_ctrlrange[aid]
        goal_ctrl[aid] = np.clip(float(value), low, high)

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

        if realtime and viewer is not None:
            sleep_time = model.opt.timestep - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    return True


def close_left_gripper_slowly(model, data, duration=2.5, viewer=None, realtime=False):
    aid = actuator_id(model, "left_finger1_ctrl")
    low, high = model.actuator_ctrlrange[aid]

    start_value = float(np.clip(LEFT_FINGER_PRE_OPEN, low, high))
    end_value = float(np.clip(LEFT_FINGER_CLOSE, low, high))

    print(f"[LEFT] close gripper: {start_value:.4f} -> {end_value:.4f}")

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

        if realtime and viewer is not None:
            sleep_time = model.opt.timestep - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    return True


def reset_scene(model, data, cube_pos, viewer=None, realtime=False):
    load_home_quiet(model, data)

    set_free_body_pos(model, data, "orange_cube", cube_pos)

    set_ctrl(model, data, "left_finger1_ctrl", LEFT_FINGER_PRE_OPEN)
    set_ctrl(model, data, "right_finger1_ctrl", right_rule.RIGHT_FINGER_PRE_OPEN)
    set_ctrl(model, data, "lifter_ctrl", 0.0)

    mujoco.mj_forward(model, data)

    if viewer is not None:
        viewer.sync()

    sim_steps(model, data, steps=700, viewer=viewer, realtime=realtime)

    # 稳定后重新放回，保证每个 case 一致
    set_free_body_pos(model, data, "orange_cube", cube_pos)

    set_ctrl(model, data, "left_finger1_ctrl", LEFT_FINGER_PRE_OPEN)
    set_ctrl(model, data, "right_finger1_ctrl", right_rule.RIGHT_FINGER_PRE_OPEN)
    set_ctrl(model, data, "lifter_ctrl", 0.0)

    mujoco.mj_forward(model, data)

    if viewer is not None:
        viewer.sync()

    sim_steps(model, data, steps=180, viewer=viewer, realtime=realtime)


# ============================================================
# Selector
# ============================================================

def select_arm(cube_pos):
    if float(cube_pos[1]) >= SELECT_RIGHT_IF_CUBE_Y_GE:
        return "right"

    return "left"


# ============================================================
# 左臂 rule trial
# ============================================================

def solve_left_ik(model, data, target_pos):
    if not hasattr(left_rule, "solve_left_arm_ik"):
        raise RuntimeError("right_pick_place.py 里找不到 solve_left_arm_ik 函数。")

    # 不同版本 solve_left_arm_ik 的参数可能不同，这里做兼容。
    try:
        result = left_rule.solve_left_arm_ik(
            model,
            data,
            target_pos,
            site_name=LEFT_SITE_NAME,
        )
    except TypeError:
        result = left_rule.solve_left_arm_ik(
            model,
            data,
            target_pos,
        )

    if not isinstance(result, tuple):
        raise RuntimeError(f"solve_left_arm_ik 返回值不是 tuple：{result}")

    if len(result) == 2:
        success, ctrl_targets = result
        return bool(success), ctrl_targets

    if len(result) >= 4:
        success = result[0]
        ctrl_targets = result[-1]
        return bool(success), ctrl_targets

    raise RuntimeError(f"无法识别 solve_left_arm_ik 返回值：{result}")


def ik_move_left_to(model, data, target_pos, label, duration, viewer=None, realtime=False):
    print("")
    print(f"[LEFT] IK move: {label}")
    print("[LEFT] target:", target_pos)

    success, ctrl_targets = solve_left_ik(model, data, target_pos)
    
        # 左臂 pick 阶段加 joint7 镜像偏置，对齐右臂成功配置里的 right_joint7_ctrl=-0.060
    if label in ["pregrasp", "grasp"]:
        for bias_name, bias_value in LEFT_PICK_JOINT_BIASES.items():
            if bias_name in ctrl_targets:
                old_value = ctrl_targets[bias_name]
                ctrl_targets[bias_name] = old_value + bias_value
                print(
                    f"[LEFT BIAS] {bias_name}: "
                    f"{old_value:.6f} -> {ctrl_targets[bias_name]:.6f}"
                )

    if not success:
        print(f"[LEFT] warning: IK not fully converged for {label}, still moving.")

    move_to_ctrl(
        model=model,
        data=data,
        targets=ctrl_targets,
        duration=duration,
        viewer=viewer,
        realtime=realtime,
    )

    sim_steps(model, data, steps=180, viewer=viewer, realtime=realtime)

    actual = get_site_pos(model, data, LEFT_SITE_NAME)
    err = float(np.linalg.norm(np.asarray(target_pos, dtype=float) - actual))

    print("[LEFT] actual:", actual)
    print("[LEFT] err:", err)

    return success, err


def run_left_rule_trial(model, data, cube_pos, viewer=None, realtime=False):
    print("")
    print("=" * 80)
    print("LEFT rule trial")
    print("=" * 80)
    print("cube_pos:", cube_pos)
    print("LEFT_SITE_NAME:", LEFT_SITE_NAME)
    print("LEFT_CUBE_PREGRASP_OFFSET:", LEFT_CUBE_PREGRASP_OFFSET)
    print("LEFT_CUBE_GRASP_OFFSET:", LEFT_CUBE_GRASP_OFFSET)
    print("LEFT_FRAME_PREPLACE_OFFSET:", LEFT_FRAME_PREPLACE_OFFSET)
    print("LEFT_FRAME_PLACE_OFFSET:", LEFT_FRAME_PLACE_OFFSET)

    reset_scene(
        model=model,
        data=data,
        cube_pos=cube_pos,
        viewer=viewer,
        realtime=realtime,
    )

    cube_initial = get_body_pos(model, data, "orange_cube")
    cube_initial_z = float(cube_initial[2])

    print("[LEFT] stable cube initial:", cube_initial)

    # 1. 左夹爪预张开
    print("[LEFT] 1. open/pre-open gripper")
    move_to_ctrl(
        model=model,
        data=data,
        targets={"left_finger1_ctrl": LEFT_FINGER_PRE_OPEN},
        duration=1.0,
        viewer=viewer,
        realtime=realtime,
    )
    sim_steps(model, data, steps=180, viewer=viewer, realtime=realtime)

    # 2. pregrasp
    cube_now = get_body_pos(model, data, "orange_cube")
    pregrasp_target = cube_now + LEFT_CUBE_PREGRASP_OFFSET

    ik_move_left_to(
        model=model,
        data=data,
        target_pos=pregrasp_target,
        label="pregrasp",
        duration=2.0,
        viewer=viewer,
        realtime=realtime,
    )

    # 3. grasp
    cube_now = get_body_pos(model, data, "orange_cube")
    grasp_target = cube_now + LEFT_CUBE_GRASP_OFFSET

    ik_move_left_to(
        model=model,
        data=data,
        target_pos=grasp_target,
        label="grasp",
        duration=1.5,
        viewer=viewer,
        realtime=realtime,
    )

    # 4. close
    print("[LEFT] 4. close gripper")
    close_left_gripper_slowly(
        model=model,
        data=data,
        duration=2.5,
        viewer=viewer,
        realtime=realtime,
    )
    sim_steps(model, data, steps=350, viewer=viewer, realtime=realtime)

    # 5. lift
    print("[LEFT] 5. lift")
    move_to_ctrl(
        model=model,
        data=data,
        targets={
            "lifter_ctrl": LIFTER_UP,
            "left_finger1_ctrl": LEFT_FINGER_CLOSE,
        },
        duration=1.5,
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

    cube_after_lift = get_body_pos(model, data, "orange_cube")

    lift_delta = float(max_cube_z - cube_initial_z)
    final_lift_delta = float(cube_after_lift[2] - cube_initial_z)

    pick_success = lift_delta > LIFT_SUCCESS_DELTA_Z

    print("[LEFT] lift_delta:", lift_delta)
    print("[LEFT] final_lift_delta:", final_lift_delta)
    print("[LEFT] pick_success:", pick_success)

    if pick_success:
        # 6. preplace
        frame_pos = get_body_pos(model, data, "black_frame")
        preplace_target = frame_pos + LEFT_FRAME_PREPLACE_OFFSET

        ik_move_left_to(
            model=model,
            data=data,
            target_pos=preplace_target,
            label="preplace",
            duration=2.0,
            viewer=viewer,
            realtime=realtime,
        )

        # 7. place
        frame_pos = get_body_pos(model, data, "black_frame")
        place_target = frame_pos + LEFT_FRAME_PLACE_OFFSET

        ik_move_left_to(
            model=model,
            data=data,
            target_pos=place_target,
            label="place",
            duration=1.5,
            viewer=viewer,
            realtime=realtime,
        )

        # 8. release
        print("[LEFT] 8. release")
        move_to_ctrl(
            model=model,
            data=data,
            targets={"left_finger1_ctrl": LEFT_FINGER_OPEN},
            duration=1.0,
            viewer=viewer,
            realtime=realtime,
        )

        sim_steps(model, data, steps=500, viewer=viewer, realtime=realtime)

    final_cube = get_body_pos(model, data, "orange_cube")
    frame_pos = get_body_pos(model, data, "black_frame")

    xy_dist = float(np.linalg.norm(final_cube[:2] - frame_pos[:2]))
    z_margin = float(final_cube[2] - frame_pos[2])

    place_success = bool(pick_success and xy_dist < 0.055 and z_margin > 0.005)

    return {
        "arm": "left",
        "pick_success": bool(pick_success),
        "place_success": bool(place_success),
        "lift_delta": float(lift_delta),
        "final_lift_delta": float(final_lift_delta),
        "xy_dist": float(xy_dist),
        "z_margin": float(z_margin),
        "cube_final": final_cube.copy(),
        "frame_final": frame_pos.copy(),
    }


# ============================================================
# 右臂 rule trial
# ============================================================

def run_right_rule_trial(model, data, cube_pos, viewer=None, realtime=False):
    print("")
    print("=" * 80)
    print("RIGHT rule trial")
    print("=" * 80)
    print("cube_pos:", cube_pos)

    right_rule.FIXED_CUBE_POS = np.asarray(cube_pos, dtype=float).copy()

    site_name = right_rule.choose_right_site(model)

    print("[RIGHT] site:", site_name)
    print("[RIGHT] config:", RIGHT_BEST_CONFIG["name"])

    result = right_rule.run_trial(
        model=model,
        site_name=site_name,
        config=RIGHT_BEST_CONFIG,
        do_place=True,
        viewer=viewer,
        realtime=realtime,
        data=data,
    )

    return {
        "arm": "right",
        "pick_success": bool(result["pick_success"]),
        "place_success": bool(result["place_success"]),
        "lift_delta": float(result["lift_delta"]),
        "final_lift_delta": float(result["final_lift_delta"]),
        "xy_dist": float(result["xy_dist"]),
        "z_margin": float(result["z_margin"]),
        "cube_final": result["cube_final"].copy(),
        "frame_final": result["frame_final"].copy(),
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("Bimanual rule selector demo")
    print("=" * 80)
    print("XML_PATH:", XML_PATH)
    print("selector: right if cube_y >=", SELECT_RIGHT_IF_CUBE_Y_GE)
    print("test cases:")
    for c in TEST_CASES:
        print(" ", c["name"], c["cube_pos"])
    print("=" * 80)

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    results = []

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()

        print("viewer 已打开。")
        print("2 秒后开始双臂 selector demo。")
        time.sleep(2)

        for i, case in enumerate(TEST_CASES, start=1):
            name = case["name"]
            cube_pos = np.asarray(case["cube_pos"], dtype=float)

            arm = select_arm(cube_pos)

            print("")
            print("#" * 80)
            print(f"CASE {i}/{len(TEST_CASES)}: {name}")
            print("cube_pos:", cube_pos)
            print("selected arm:", arm)
            print("#" * 80)

            if arm == "right":
                result = run_right_rule_trial(
                    model=model,
                    data=data,
                    cube_pos=cube_pos,
                    viewer=viewer,
                    realtime=True,
                )
            else:
                result = run_left_rule_trial(
                    model=model,
                    data=data,
                    cube_pos=cube_pos,
                    viewer=viewer,
                    realtime=True,
                )

            result["case"] = name
            result["cube_initial"] = cube_pos.copy()

            results.append(result)

            print("")
            print("=" * 80)
            print(f"CASE RESULT: {name}")
            print("=" * 80)
            print("arm:", result["arm"])
            print("pick_success:", result["pick_success"])
            print("place_success:", result["place_success"])
            print("lift_delta:", result["lift_delta"])
            print("final_lift_delta:", result["final_lift_delta"])
            print("xy_dist:", result["xy_dist"])
            print("z_margin:", result["z_margin"])
            print("cube_final:", result["cube_final"])
            print("frame_final:", result["frame_final"])

            print("")
            print("暂停 2 秒进入下一个 case。")
            sim_steps(
                model,
                data,
                steps=int(2.0 / model.opt.timestep),
                viewer=viewer,
                realtime=True,
            )

        print("")
        print("=" * 80)
        print("Bimanual selector demo 总结")
        print("=" * 80)

        success_count = 0

        for r in results:
            if r["place_success"]:
                success_count += 1

            print(
                f"{r['case']:12s} "
                f"arm={r['arm']:5s} "
                f"pick={r['pick_success']} "
                f"place={r['place_success']} "
                f"lift={r['lift_delta']:.4f} "
                f"xy={r['xy_dist']:.4f} "
                f"z_margin={r['z_margin']:.4f}"
            )

        print("")
        print(f"total place_success: {success_count}/{len(results)}")

        print("")
        print("viewer 保持运行。关闭 viewer 或 Ctrl+C 结束。")

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


# 当前文件位置：
# E:\FL_Personal\openarm_mujoco-master\openarm_mujoco-master\scripts\right_pick_place.py
# parents[1] 是项目根目录：
# E:\FL_Personal\openarm_mujoco-master\openarm_mujoco-master
ROOT = Path(__file__).resolve().parents[1]
XML_PATH = ROOT / "v2" / "demo.xml"


# =========================
# 你记录的关键参数
# =========================

LEFT_JOINT3_GRASP = 0.43       # 你记录的是 0.424 ~ 0.44，这里取中间值
LEFT_FINGER_OPEN_1 = 0.448
LEFT_JOINT1_CUBE = 0.223
LEFT_FINGER_CLOSE = 0.0
LEFT_JOINT1_BACK = 0.0
LIFTER_UP = 0.11
LEFT_JOINT3_PLACE = 0.0314
LEFT_JOINT1_FRAME = 0.15
LEFT_FINGER_RELEASE = 0.49


# 方块抬起高度判断阈值
# 程序会记录方块初始高度，如果之后 z 增加超过这个值，就认为抓取成功
LIFT_SUCCESS_DELTA_Z = 0.025


def get_id(model, obj_type, name):
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id == -1:
        raise ValueError(f"找不到对象：{name}")
    return obj_id


def actuator_id(model, name):
    return get_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def body_id(model, name):
    return get_id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def get_body_pos(model, data, name):
    bid = body_id(model, name)
    return data.xpos[bid].copy()


def get_cube_pos(model, data):
    return get_body_pos(model, data, "orange_cube")


def set_ctrl(model, data, name, value):
    """
    设置某个 actuator 的 ctrl 值。
    """
    aid = actuator_id(model, name)
    low, high = model.actuator_ctrlrange[aid]
    data.ctrl[aid] = np.clip(value, low, high)


def move_to(model, data, viewer, targets, duration=1.5):
    """
    平滑移动到目标控制量。
    targets 示例：
    {
        "left_joint1_ctrl": 0.223,
        "left_joint3_ctrl": 0.43,
    }
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

        elapsed = time.time() - step_start
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


def hold(model, data, viewer, duration=0.5):
    """
    保持当前控制量一段时间。
    """
    dt = model.opt.timestep
    steps = max(1, int(duration / dt))

    for _ in range(steps):
        if not viewer.is_running():
            return

        step_start = time.time()

        mujoco.mj_step(model, data)
        viewer.sync()

        elapsed = time.time() - step_start
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


def close_gripper_slowly(
    model,
    data,
    viewer,
    actuator_name="left_finger1_ctrl",
    start_value=LEFT_FINGER_OPEN_1,
    end_value=LEFT_FINGER_CLOSE,
    duration=1.5,
):
    """
    自动夹爪闭合逻辑：
    从张开值慢慢闭合到关闭值，而不是一下子跳到关闭。
    """
    aid = actuator_id(model, actuator_name)
    low, high = model.actuator_ctrlrange[aid]

    start_value = float(np.clip(start_value, low, high))
    end_value = float(np.clip(end_value, low, high))

    dt = model.opt.timestep
    steps = max(1, int(duration / dt))

    print(f"自动闭合夹爪：{start_value:.4f} -> {end_value:.4f}")

    for i in range(steps):
        if not viewer.is_running():
            return

        alpha = (i + 1) / steps
        alpha = 3 * alpha ** 2 - 2 * alpha ** 3

        value = (1 - alpha) * start_value + alpha * end_value
        data.ctrl[aid] = value

        step_start = time.time()

        mujoco.mj_step(model, data)
        viewer.sync()

        elapsed = time.time() - step_start
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


def is_cube_lifted(model, data, initial_cube_z, delta_z=LIFT_SUCCESS_DELTA_Z):
    """
    判断方块是否被抬起。
    """
    cube_pos = get_cube_pos(model, data)
    current_z = cube_pos[2]
    lifted = current_z > initial_cube_z + delta_z

    print(
        f"方块高度检测：initial_z={initial_cube_z:.4f}, "
        f"current_z={current_z:.4f}, "
        f"delta={current_z - initial_cube_z:.4f}"
    )

    return lifted


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

    # 把每个 joint actuator 的 ctrl 初始化为当前 qpos
    for aid in range(model.nu):
        joint_id = model.actuator_trnid[aid, 0]

        if joint_id < 0:
            continue

        qpos_addr = model.jnt_qposadr[joint_id]

        low, high = model.actuator_ctrlrange[aid]
        data.ctrl[aid] = np.clip(data.qpos[qpos_addr], low, high)

    mujoco.mj_forward(model, data)


def print_actuators(model):
    """
    打印所有 actuator 名字和范围，方便检查 ctrl 名字是否正确。
    """
    print("\n========== Actuators ==========")

    for aid in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        low, high = model.actuator_ctrlrange[aid]
        print(f"{aid:02d}  {name:30s}  ctrlrange=[{low:.4f}, {high:.4f}]")

    print("================================\n")


def main():
    print(f"加载模型：{XML_PATH}")

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    load_home(model, data)
    print_actuators(model)

    # 初始化后记录方块初始高度
    cube_initial_pos = get_cube_pos(model, data)
    cube_initial_z = cube_initial_pos[2]

    print(f"orange_cube 初始位置：{cube_initial_pos}")
    print(f"orange_cube 初始高度 z：{cube_initial_z:.4f}")

    # =========================
    # 左臂动作参数
    # =========================

    step1_joint3_to_grasp = {
        "left_joint3_ctrl": LEFT_JOINT3_GRASP,
    }

    step2_open_gripper = {
        "left_finger1_ctrl": LEFT_FINGER_OPEN_1,
    }

    step3_joint1_to_cube = {
        "left_joint1_ctrl": LEFT_JOINT1_CUBE,
    }

    step5_joint1_back = {
        "left_joint1_ctrl": LEFT_JOINT1_BACK,
    }

    step6_lifter_up = {
        "lifter_ctrl": LIFTER_UP,
    }

    step7_joint3_to_place = {
        "left_joint3_ctrl": LEFT_JOINT3_PLACE,
    }

    step8_joint1_to_frame = {
        "left_joint1_ctrl": LEFT_JOINT1_FRAME,
    }

    step9_release_gripper = {
        "left_finger1_ctrl": LEFT_FINGER_RELEASE,
    }

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()

        print("viewer 已打开。")
        print("3 秒后开始执行左臂动作。")
        time.sleep(3)

        print("1. left_joint3 到抓取前姿态")
        move_to(model, data, viewer, step1_joint3_to_grasp, duration=1.5)
        hold(model, data, viewer, duration=0.5)

        print("2. 张开左夹爪")
        move_to(model, data, viewer, step2_open_gripper, duration=1.0)
        hold(model, data, viewer, duration=0.5)

        print("3. left_joint1 转到方块位置")
        move_to(model, data, viewer, step3_joint1_to_cube, duration=1.5)
        hold(model, data, viewer, duration=0.5)

        print("4. 自动慢慢闭合左夹爪，尝试夹住方块")
        close_gripper_slowly(
            model,
            data,
            viewer,
            actuator_name="left_finger1_ctrl",
            start_value=LEFT_FINGER_OPEN_1,
            end_value=LEFT_FINGER_CLOSE,
            duration=1.5,
        )
        hold(model, data, viewer, duration=1.0)

        print("5. left_joint1 回到 0")
        move_to(model, data, viewer, step5_joint1_back, duration=1.5)
        hold(model, data, viewer, duration=0.5)

        print("6. lifter 抬高")
        move_to(model, data, viewer, step6_lifter_up, duration=1.5)
        hold(model, data, viewer, duration=1.0)

        print("7. 检测是否抓取成功")
        success = is_cube_lifted(model, data, cube_initial_z)

        if not success:
            print("抓取失败：方块没有明显被抬起。")
            print("后续放置动作停止。请检查夹爪是否夹住方块，或者调大/调小抓取参数。")

            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)

            return

        print("抓取成功：方块已经被抬起，继续执行放置动作。")

        print("8. left_joint3 到放置前姿态")
        move_to(model, data, viewer, step7_joint3_to_place, duration=1.5)
        hold(model, data, viewer, duration=0.5)

        print("9. left_joint1 转到黑色框架位置")
        move_to(model, data, viewer, step8_joint1_to_frame, duration=1.5)
        hold(model, data, viewer, duration=0.5)

        print("10. 张开左夹爪释放")
        move_to(model, data, viewer, step9_release_gripper, duration=1.0)
        hold(model, data, viewer, duration=0.5)

        print("左臂抓取 + 检测 + 放置流程执行完毕。viewer 保持运行。")

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
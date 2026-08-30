from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


# 当前文件在 scripts/scripted_pick.py
# parents[1] 就是项目根目录：
# E:\FL_Personal\openarm_mujoco-master\openarm_mujoco-master
ROOT = Path(__file__).resolve().parents[1]
XML_PATH = ROOT / "v2" / "demo.xml"


def get_id(model, obj_type, name):
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id == -1:
        raise ValueError(f"找不到对象：{name}")
    return obj_id


def actuator_id(model, name):
    return get_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def set_ctrl(model, data, name, value):
    """
    设置某个 actuator 的 ctrl 值。
    """
    aid = actuator_id(model, name)

    low, high = model.actuator_ctrlrange[aid]
    value = float(np.clip(value, low, high))

    data.ctrl[aid] = value


def set_many_ctrl(model, data, targets):
    """
    一次设置多个 actuator。
    targets 示例：
    {
        "left_joint1_ctrl": 0.2,
        "left_joint2_ctrl": -0.5,
    }
    """
    for name, value in targets.items():
        set_ctrl(model, data, name, value)


def move_to(model, data, viewer, targets, duration=2.0):
    """
    平滑移动到目标控制量。
    targets 是 actuator_name -> target_value 的字典。
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

        # smoothstep，避免动作太突兀
        alpha = 3 * alpha ** 2 - 2 * alpha ** 3

        data.ctrl[:] = (1 - alpha) * start_ctrl + alpha * goal_ctrl

        step_start = time.time()

        mujoco.mj_step(model, data)
        viewer.sync()

        elapsed = time.time() - step_start
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


def hold(model, data, viewer, duration=1.0):
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


def load_home(model, data):
    """
    加载 XML 里定义的 home keyframe。
    然后把 position actuator 的 ctrl 同步到当前 qpos，
    这样机器人不会一启动就往零位姿态掉。
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
    打印所有 actuator 名字和控制范围。
    后面你调动作时会用到这些名字。
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

    # 下面是一个非常基础的 scripted grasp 示例。
    # 这些数值只是起点，不保证一次就抓住，需要你根据画面微调。

    open_gripper = {
        # 一般较大是张开，较小是闭合。
        # 具体范围以终端打印的 ctrlrange 为准。
        "left_finger1_ctrl": 0.04,
    }

    pre_grasp = {
        # 左臂移动到方块附近的一个大概姿态
        "left_joint1_ctrl": 0.25,
        "left_joint2_ctrl": -0.55,
        "left_joint3_ctrl": 0.00,
        "left_joint4_ctrl": 1.25,
        "left_joint5_ctrl": 0.00,
        "left_joint6_ctrl": 0.55,
        "left_joint7_ctrl": 0.00,
    }

    lower = {
        # 再往下/往前一点，尝试接近方块
        "left_joint2_ctrl": -0.45,
        "left_joint4_ctrl": 1.55,
        "left_joint6_ctrl": 0.35,
    }

    close_gripper = {
        # 闭合夹爪
        "left_finger1_ctrl": 0.00,
    }

    lift = {
        # 抬起来
        "left_joint2_ctrl": -0.85,
        "left_joint4_ctrl": 1.05,
        "left_joint6_ctrl": 0.75,
    }

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()

        print("viewer 已打开。")
        print("3 秒后开始执行 scripted grasp。")
        time.sleep(3)

        print("1. 张开夹爪")
        move_to(model, data, viewer, open_gripper, duration=1.0)
        hold(model, data, viewer, duration=0.5)

        print("2. 移动到方块附近")
        move_to(model, data, viewer, pre_grasp, duration=3.0)
        hold(model, data, viewer, duration=0.5)

        print("3. 下探")
        move_to(model, data, viewer, lower, duration=2.0)
        hold(model, data, viewer, duration=0.5)

        print("4. 闭合夹爪")
        move_to(model, data, viewer, close_gripper, duration=1.0)
        hold(model, data, viewer, duration=1.0)

        print("5. 抬起")
        move_to(model, data, viewer, lift, duration=2.0)

        print("动作执行完毕。viewer 保持运行。")

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
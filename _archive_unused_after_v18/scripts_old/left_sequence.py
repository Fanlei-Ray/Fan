from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
XML_PATH = ROOT / "v2" / "demo.xml"


def actuator_id(model, name):
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid == -1:
        raise ValueError(f"找不到 actuator: {name}")
    return aid


def move_to(model, data, viewer, targets, duration=1.5):
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
        alpha = 3 * alpha**2 - 2 * alpha**3

        data.ctrl[:] = (1 - alpha) * start_ctrl + alpha * goal_ctrl

        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()

        sleep_time = dt - (time.time() - step_start)
        if sleep_time > 0:
            time.sleep(sleep_time)


def hold(model, data, viewer, duration=0.5):
    dt = model.opt.timestep
    steps = max(1, int(duration / dt))

    for _ in range(steps):
        if not viewer.is_running():
            return

        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()

        sleep_time = dt - (time.time() - step_start)
        if sleep_time > 0:
            time.sleep(sleep_time)


def load_home(model, data):
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        print("已加载 home keyframe。")
    else:
        print("没有找到 home keyframe。")

    # 把 position actuator 的 ctrl 初始化到当前关节位置
    for aid in range(model.nu):
        joint_id = model.actuator_trnid[aid, 0]
        if joint_id < 0:
            continue

        qpos_addr = model.jnt_qposadr[joint_id]
        low, high = model.actuator_ctrlrange[aid]
        data.ctrl[aid] = np.clip(data.qpos[qpos_addr], low, high)

    mujoco.mj_forward(model, data)


def print_actuators(model):
    print("\n========== Actuators ==========")
    for aid in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        low, high = model.actuator_ctrlrange[aid]
        print(f"{aid:02d}  {name:30s}  ctrlrange=[{low:.4f}, {high:.4f}]")
    print("================================\n")


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    load_home(model, data)
    print_actuators(model)

    # 根据你记录的左臂关键参数写成顺序动作
    sequence = [
        ("1. 调 left_joint3 到抓取前姿态", {
            "left_joint3_ctrl": 0.43,
        }, 1.5),

        ("2. 调 left_finger1，准备夹爪", {
            "left_finger1_ctrl": 0.448,
        }, 1.0),

        ("3. 调 left_joint1 到方块位置", {
            "left_joint1_ctrl": 0.223,
        }, 1.5),

        ("4. 闭合/夹住", {
            "left_finger1_ctrl": 0.0,
        }, 1.0),

        ("5. left_joint1 回到 0", {
            "left_joint1_ctrl": 0.0,
        }, 1.5),

        ("6. 升降台抬高", {
            "lifter_ctrl": 0.11,
        }, 1.5),

        ("7. 调 left_joint3 到放置前姿态", {
            "left_joint3_ctrl": 0.0314,
        }, 1.5),

        ("8. 调 left_joint1 到黑框位置", {
            "left_joint1_ctrl": 0.15,
        }, 1.5),

        ("9. 张开夹爪释放", {
            "left_finger1_ctrl": 0.49,
        }, 1.0),
    ]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()

        print("viewer 已打开，3 秒后开始左臂动作。")
        time.sleep(3)

        for name, targets, duration in sequence:
            print(name, targets)
            move_to(model, data, viewer, targets, duration=duration)
            hold(model, data, viewer, duration=0.5)

        print("动作执行完毕，viewer 保持运行。")

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
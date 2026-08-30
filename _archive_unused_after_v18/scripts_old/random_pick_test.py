from pathlib import Path
import time
import random

import mujoco
import mujoco.viewer
import numpy as np

# 直接复用你已经跑通的任务式脚本
import right_pick_place as pp


# ============================================================
# 随机测试参数
# ============================================================

NUM_TRIALS = 5

# 先用小范围，确认稳定后再扩大
CUBE_X_RANGE = (0.49, 0.55)
CUBE_Y_RANGE = (-0.04, 0.04)
CUBE_Z = 1.05

# 是否固定随机种子
# 固定后每次随机结果一样，方便复现问题
RANDOM_SEED = 42


# ============================================================
# 工具函数
# ============================================================

def find_freejoint_for_body(model, body_name):
    """
    找到某个 body 对应的 freejoint。
    demo.xml 里的 orange_cube 有 <freejoint/>，但这个 joint 不一定有名字，
    所以不能只靠 joint name，要从 body 反查 joint。
    """
    body_id = pp.body_id(model, body_name)

    for jid in range(model.njnt):
        if model.jnt_bodyid[jid] == body_id:
            joint_type = model.jnt_type[jid]

            if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                return jid

    raise ValueError(f"{body_name} 没有找到 freejoint")


def set_free_body_pose(model, data, body_name, pos, quat=None):
    """
    设置带 freejoint 的 body 的世界位置。
    freejoint 的 qpos 格式是：
        [x, y, z, qw, qx, qy, qz]
    """
    if quat is None:
        quat = np.array([1.0, 0.0, 0.0, 0.0])

    jid = find_freejoint_for_body(model, body_name)

    qaddr = model.jnt_qposadr[jid]
    daddr = model.jnt_dofadr[jid]

    data.qpos[qaddr:qaddr + 3] = np.asarray(pos, dtype=float)
    data.qpos[qaddr + 3:qaddr + 7] = np.asarray(quat, dtype=float)

    # 清零这个 freejoint 的速度，防止方块一开始带速度乱飞
    data.qvel[daddr:daddr + 6] = 0.0

    mujoco.mj_forward(model, data)


def sample_cube_position():
    x = random.uniform(*CUBE_X_RANGE)
    y = random.uniform(*CUBE_Y_RANGE)
    z = CUBE_Z
    return np.array([x, y, z], dtype=float)


def reset_scene_for_trial(model, data, cube_pos):
    """
    每次 trial 前：
    1. 机器人回 home
    2. actuator ctrl 同步
    3. 方块放到随机位置
    """
    pp.load_home(model, data)

    set_free_body_pose(
        model,
        data,
        pp.PICK_BODY_NAME,
        cube_pos,
    )

    print("\n--------------------------------------------------")
    print("已重置场景")
    print(f"{pp.PICK_BODY_NAME} 随机位置：{cube_pos}")
    print(f"{pp.PLACE_BODY_NAME} 当前位置：{pp.get_body_pos(model, data, pp.PLACE_BODY_NAME)}")
    print(f"{pp.LEFT_SITE_NAME} 当前位置：{pp.get_site_pos(model, data, pp.LEFT_SITE_NAME)}")
    print("--------------------------------------------------")


# ============================================================
# 主程序
# ============================================================

def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    print(f"加载模型：{pp.XML_PATH}")

    model = mujoco.MjModel.from_xml_path(str(pp.XML_PATH))
    data = mujoco.MjData(model)

    pp.load_home(model, data)
    pp.print_actuators(model)

    print("\n随机测试配置：")
    print(f"NUM_TRIALS = {NUM_TRIALS}")
    print(f"CUBE_X_RANGE = {CUBE_X_RANGE}")
    print(f"CUBE_Y_RANGE = {CUBE_Y_RANGE}")
    print(f"CUBE_Z = {CUBE_Z}")

    results = []

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()

        print("\nviewer 已打开。")
        print("3 秒后开始随机位置 pick-and-place 测试。")
        time.sleep(3)

        for trial_idx in range(1, NUM_TRIALS + 1):
            if not viewer.is_running():
                break

            print("\n\n##################################################")
            print(f"随机测试 Trial {trial_idx}/{NUM_TRIALS}")
            print("##################################################")

            cube_pos = sample_cube_position()
            reset_scene_for_trial(model, data, cube_pos)

            # 稍微等一下，让你能看到随机位置
            pp.hold(model, data, viewer, duration=1.0)

            success = pp.run_pick_and_place_task(model, data, viewer)

            results.append(
                {
                    "trial": trial_idx,
                    "cube_pos": cube_pos.copy(),
                    "success": bool(success),
                }
            )

            print("\nTrial 结果：")
            print(f"trial = {trial_idx}")
            print(f"cube_pos = {cube_pos}")
            print(f"success = {success}")

            # 每次 trial 之间暂停一下
            pp.hold(model, data, viewer, duration=2.0)

        print("\n\n==================================================")
        print("随机测试总结")
        print("==================================================")

        success_count = sum(1 for r in results if r["success"])
        total_count = len(results)

        for r in results:
            status = "成功" if r["success"] else "失败"
            print(
                f"Trial {r['trial']:02d}: {status}, "
                f"cube_pos={r['cube_pos']}"
            )

        if total_count > 0:
            rate = success_count / total_count
            print(f"\n成功次数：{success_count}/{total_count}")
            print(f"成功率：{rate * 100:.1f}%")
        else:
            print("没有完成任何 trial。")

        print("\n测试结束，viewer 保持运行。")

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
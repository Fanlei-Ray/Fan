from pathlib import Path
import numpy as np

from openarm_pickplace_env import OpenArmPickPlaceEnv
from collect_bc_data_v5 import ExpertPhasePolicy


ROOT = Path(__file__).resolve().parents[1]

X_VALUES = np.round(np.linspace(0.46, 0.59, 8), 3)
Y_VALUES = np.round(np.linspace(-0.10, 0.10, 9), 3)

MAX_EPISODE_STEPS = 460

FRAME_SKIP = 30
MAX_TCP_DELTA = 0.024
MAX_FINGER_DELTA = 0.040
MAX_LIFTER_DELTA = 0.025


def make_env(x, y):
    env = OpenArmPickPlaceEnv(
        render_mode=None,
        randomize_cube=True,
        cube_x_range=(float(x), float(x)),
        cube_y_range=(float(y), float(y)),
        max_episode_steps=MAX_EPISODE_STEPS,
    )

    env.frame_skip = FRAME_SKIP
    env.max_tcp_delta = MAX_TCP_DELTA
    env.max_finger_delta = MAX_FINGER_DELTA
    env.max_lifter_delta = MAX_LIFTER_DELTA

    return env


def run_one_position(x, y, seed):
    env = make_env(x, y)
    expert = ExpertPhasePolicy()

    obs, info = env.reset(seed=seed)
    expert.reset(info)

    final_info = info
    finished = False

    for step in range(MAX_EPISODE_STEPS):
        action = expert.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        final_info = info

        expert.update(obs, info)

        if expert.phase == "done_hold" and expert.phase_steps > 8:
            finished = True
            break

        if truncated:
            break

    success = bool(final_info["is_success"]) and finished

    env.close()

    return success, expert.phase, final_info["cube_pos"]


def main():
    print("开始扫描左臂工作空间")
    print("X_VALUES:", X_VALUES)
    print("Y_VALUES:", Y_VALUES)
    print("")
    print("图例：")
    print("  O = 左臂 expert 成功")
    print("  X = 左臂 expert 失败")
    print("")

    results = {}

    for y in reversed(Y_VALUES):
        row = []

        for x in X_VALUES:
            seed = int(90000 + round(x * 1000) * 10 + round((y + 1.0) * 1000))

            success, phase, cube_pos = run_one_position(x, y, seed)

            results[(float(x), float(y))] = success

            mark = "O" if success else "X"
            row.append(mark)

            print(
                f"x={x:.3f}, y={y:.3f} -> {mark}, "
                f"final_phase={phase}, "
                f"cube={np.array2string(cube_pos, precision=3)}"
            )

        print("")

    print("")
    print("=" * 70)
    print("左臂工作空间 ASCII 图")
    print("=" * 70)
    print("列是 x：", " ".join([f"{x:.3f}" for x in X_VALUES]))
    print("行是 y，从 +y 到 -y")
    print("")

    for y in reversed(Y_VALUES):
        row = []
        for x in X_VALUES:
            row.append("O" if results[(float(x), float(y))] else "X")

        print(f"y={y:+.3f}:  " + "   ".join(row))

    print("")
    print("建议：只用 O 连续成片的区域训练左臂。")
    print("如果某些区域左臂全是 X，那里应该交给右臂，或者不要放入左臂训练范围。")


if __name__ == "__main__":
    main()
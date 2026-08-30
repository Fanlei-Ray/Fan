from pathlib import Path
import sys

import mujoco
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import right_rule_pick_place as rr


NUM_EPISODES = 30

# 先只测右臂固定点附近小范围，不要一上来全桌面
CUBE_X_RANGE = (0.49, 0.55)
CUBE_Y_RANGE = (0.02, 0.08)
CUBE_Z = 1.050

SEED_START = 30001


BEST_CONFIG = {
    "name": "right_best_tcp_x-0.006_y-0.020_z+0.055__j7_m060",
    "site_type": "tcp",

    # 来自刚才成功配置
    "pregrasp_offset": np.array([-0.006, -0.020, 0.140], dtype=float),
    "grasp_offset": np.array([-0.006, -0.020, 0.055], dtype=float),

    # 放置保持刚才成功脚本里的默认值
    "preplace_offset": np.array([0.0, 0.0, 0.140], dtype=float),
    "place_offset": np.array([0.0, 0.0, 0.080], dtype=float),

    "joint_biases": {
        "right_joint7_ctrl": -0.060,
    },
}


def sample_cube_pos(rng):
    x = rng.uniform(CUBE_X_RANGE[0], CUBE_X_RANGE[1])
    y = rng.uniform(CUBE_Y_RANGE[0], CUBE_Y_RANGE[1])
    return np.array([x, y, CUBE_Z], dtype=float)


def main():
    print("=" * 80)
    print("Right rule expert random test")
    print("=" * 80)
    print("XML:", rr.XML_PATH)
    print("NUM_EPISODES:", NUM_EPISODES)
    print("CUBE_X_RANGE:", CUBE_X_RANGE)
    print("CUBE_Y_RANGE:", CUBE_Y_RANGE)
    print("BEST_CONFIG:", BEST_CONFIG["name"])
    print("=" * 80)

    model = mujoco.MjModel.from_xml_path(str(rr.XML_PATH))
    site_name = rr.choose_right_site(model)

    print("使用右臂 site:", site_name)

    pick_success_count = 0
    place_success_count = 0

    records = []

    for ep in range(1, NUM_EPISODES + 1):
        seed = SEED_START + ep
        rng = np.random.default_rng(seed)

        cube_pos = sample_cube_pos(rng)

        # 关键：复用 right_rule_pick_place.py 的 run_trial，
        # 但每个 episode 前改它的全局 FIXED_CUBE_POS。
        rr.FIXED_CUBE_POS = cube_pos.copy()

        result = rr.run_trial(
            model=model,
            site_name=site_name,
            config=BEST_CONFIG,
            do_place=True,
            viewer=None,
            realtime=False,
            data=None,
        )

        pick_success = bool(result["pick_success"])
        place_success = bool(result["place_success"])

        pick_success_count += int(pick_success)
        place_success_count += int(place_success)

        record = {
            "ep": ep,
            "seed": seed,
            "cube_x": float(cube_pos[0]),
            "cube_y": float(cube_pos[1]),
            "pick_success": pick_success,
            "place_success": place_success,
            "lift_delta": float(result["lift_delta"]),
            "final_lift_delta": float(result["final_lift_delta"]),
            "xy_dist": float(result["xy_dist"]),
            "z_margin": float(result["z_margin"]),
            "cube_final": result["cube_final"].copy(),
            "frame_final": result["frame_final"].copy(),
        }

        records.append(record)

        mark = "O" if place_success else ("P" if pick_success else "X")

        print(
            f"{mark} ep={ep:02d}/{NUM_EPISODES} "
            f"seed={seed} "
            f"cube=({cube_pos[0]:.3f},{cube_pos[1]:.3f}) "
            f"pick={pick_success} "
            f"place={place_success} "
            f"lift={result['lift_delta']:.4f} "
            f"xy={result['xy_dist']:.4f} "
            f"z_margin={result['z_margin']:.4f} "
            f"cube_final={np.array2string(result['cube_final'], precision=4)}"
        )

    print("")
    print("=" * 80)
    print("Right rule expert random test 总结")
    print("=" * 80)
    print(f"pick_success：{pick_success_count}/{NUM_EPISODES}")
    print(f"pick_success 成功率：{pick_success_count / NUM_EPISODES * 100:.1f}%")
    print(f"place_success：{place_success_count}/{NUM_EPISODES}")
    print(f"place_success 成功率：{place_success_count / NUM_EPISODES * 100:.1f}%")

    print("")
    print("失败样本：")
    for r in records:
        if not r["place_success"]:
            print(
                f"seed={r['seed']} "
                f"cube=({r['cube_x']:.3f},{r['cube_y']:.3f}) "
                f"pick={r['pick_success']} "
                f"place={r['place_success']} "
                f"lift={r['lift_delta']:.4f} "
                f"xy={r['xy_dist']:.4f} "
                f"z_margin={r['z_margin']:.4f}"
            )


if __name__ == "__main__":
    main()
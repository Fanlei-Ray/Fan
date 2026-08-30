from pathlib import Path
import sys
import csv
import json
import time

import mujoco
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import right_rule_pick_place as rr


# ============================================================
# Right arm success demo collector
#
# 当前阶段目标：
#   不训练。
#   不再大范围 sweep。
#   只用已经成功过的右臂 rule expert 配置采成功样本。
#
# 重要：
#   每成功 1 个 episode 就保存一次 npz。
#   每尝试 1 次就写 csv log。
#   Ctrl+C 中断也不会白跑。
# ============================================================


RESULT_DIR = ROOT / "outputs" / "right_bc_v1"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_NPZ = RESULT_DIR / "right_success_trials_v1.npz"
OUTPUT_CSV = RESULT_DIR / "right_success_trials_v1_log.csv"
TARGET_SUCCESS = 120
MAX_ATTEMPTS = 500

SEED_START = 40001

# 先采右臂小范围。
# 当前 rule expert 在这个范围约 63.3%，够采第一批成功样本。
CUBE_X_RANGE = (0.49, 0.55)
CUBE_Y_RANGE = (0.02, 0.08)
CUBE_Z = 1.050


BEST_CONFIG = {
    "name": "right_best_tcp_x-0.006_y-0.020_z+0.055__j7_m060",
    "site_type": "tcp",

    # 来自右臂 fixed pick/place 成功配置
    "pregrasp_offset": np.array([-0.006, -0.020, 0.140], dtype=float),
    "grasp_offset": np.array([-0.006, -0.020, 0.055], dtype=float),

    "preplace_offset": np.array([0.0, 0.0, 0.140], dtype=float),
    "place_offset": np.array([0.0, 0.0, 0.080], dtype=float),

    "joint_biases": {
        "right_joint7_ctrl": -0.060,
    },
}


def sample_cube_pos(seed):
    rng = np.random.default_rng(seed)

    x = rng.uniform(CUBE_X_RANGE[0], CUBE_X_RANGE[1])
    y = rng.uniform(CUBE_Y_RANGE[0], CUBE_Y_RANGE[1])

    return np.array([x, y, CUBE_Z], dtype=float)


def config_to_jsonable(config):
    out = {}

    for k, v in config.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = dict(v)
        else:
            out[k] = v

    return out


def load_existing_successes():
    if not OUTPUT_NPZ.exists():
        return []

    data = np.load(OUTPUT_NPZ, allow_pickle=True)

    successes = []

    n = len(data["seeds"])

    for i in range(n):
        successes.append(
            {
                "seed": int(data["seeds"][i]),
                "cube_pos": data["cube_positions"][i].astype(float),
                "cube_final": data["cube_finals"][i].astype(float),
                "frame_final": data["frame_finals"][i].astype(float),
                "lift_delta": float(data["lift_deltas"][i]),
                "final_lift_delta": float(data["final_lift_deltas"][i]),
                "xy_dist": float(data["xy_dists"][i]),
                "z_margin": float(data["z_margins"][i]),
            }
        )

    print(f"检测到已有成功样本：{len(successes)}，将继续追加。")

    return successes


def save_successes(successes):
    if not successes:
        return

    seeds = np.array([s["seed"] for s in successes], dtype=np.int64)
    cube_positions = np.array([s["cube_pos"] for s in successes], dtype=np.float32)
    cube_finals = np.array([s["cube_final"] for s in successes], dtype=np.float32)
    frame_finals = np.array([s["frame_final"] for s in successes], dtype=np.float32)

    lift_deltas = np.array([s["lift_delta"] for s in successes], dtype=np.float32)
    final_lift_deltas = np.array([s["final_lift_delta"] for s in successes], dtype=np.float32)
    xy_dists = np.array([s["xy_dist"] for s in successes], dtype=np.float32)
    z_margins = np.array([s["z_margin"] for s in successes], dtype=np.float32)

    config_json = json.dumps(config_to_jsonable(BEST_CONFIG), ensure_ascii=False)

    tmp_path = OUTPUT_NPZ.with_suffix(".tmp.npz")

    np.savez_compressed(
        tmp_path,
        seeds=seeds,
        cube_positions=cube_positions,
        cube_finals=cube_finals,
        frame_finals=frame_finals,
        lift_deltas=lift_deltas,
        final_lift_deltas=final_lift_deltas,
        xy_dists=xy_dists,
        z_margins=z_margins,
        config_json=np.array(config_json),
        cube_x_range=np.array(CUBE_X_RANGE, dtype=np.float32),
        cube_y_range=np.array(CUBE_Y_RANGE, dtype=np.float32),
        cube_z=np.array(CUBE_Z, dtype=np.float32),
    )

    tmp_path.replace(OUTPUT_NPZ)


def init_csv_if_needed():
    if OUTPUT_CSV.exists():
        return

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "attempt_index",
                "seed",
                "cube_x",
                "cube_y",
                "pick_success",
                "place_success",
                "lift_delta",
                "final_lift_delta",
                "xy_dist",
                "z_margin",
                "num_success_so_far",
            ]
        )


def append_attempt_log(attempt_index, seed, cube_pos, result, num_success_so_far):
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                attempt_index,
                seed,
                float(cube_pos[0]),
                float(cube_pos[1]),
                bool(result["pick_success"]),
                bool(result["place_success"]),
                float(result["lift_delta"]),
                float(result["final_lift_delta"]),
                float(result["xy_dist"]),
                float(result["z_margin"]),
                num_success_so_far,
            ]
        )


def already_collected_seed_set(successes):
    return set(int(s["seed"]) for s in successes)


def run_one_attempt(model, site_name, seed):
    cube_pos = sample_cube_pos(seed)

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

    return cube_pos, result


def main():
    print("=" * 80)
    print("Collect right-arm success trials v1")
    print("=" * 80)
    print("XML:", rr.XML_PATH)
    print("OUTPUT_NPZ:", OUTPUT_NPZ)
    print("OUTPUT_CSV:", OUTPUT_CSV)
    print("TARGET_SUCCESS:", TARGET_SUCCESS)
    print("MAX_ATTEMPTS:", MAX_ATTEMPTS)
    print("CUBE_X_RANGE:", CUBE_X_RANGE)
    print("CUBE_Y_RANGE:", CUBE_Y_RANGE)
    print("BEST_CONFIG:", BEST_CONFIG["name"])
    print("=" * 80)

    successes = load_existing_successes()
    seen_success_seeds = already_collected_seed_set(successes)

    init_csv_if_needed()

    model = mujoco.MjModel.from_xml_path(str(rr.XML_PATH))
    site_name = rr.choose_right_site(model)

    print("使用右臂 site:", site_name)

    start_time = time.time()

    attempt_index = 0
    success_before = len(successes)

    try:
        for attempt_index in range(1, MAX_ATTEMPTS + 1):
            if len(successes) >= TARGET_SUCCESS:
                print("已达到目标成功样本数。")
                break

            seed = SEED_START + attempt_index

            if seed in seen_success_seeds:
                continue

            cube_pos, result = run_one_attempt(
                model=model,
                site_name=site_name,
                seed=seed,
            )

            pick_success = bool(result["pick_success"])
            place_success = bool(result["place_success"])

            if place_success:
                item = {
                    "seed": seed,
                    "cube_pos": cube_pos.copy(),
                    "cube_final": result["cube_final"].copy(),
                    "frame_final": result["frame_final"].copy(),
                    "lift_delta": float(result["lift_delta"]),
                    "final_lift_delta": float(result["final_lift_delta"]),
                    "xy_dist": float(result["xy_dist"]),
                    "z_margin": float(result["z_margin"]),
                }

                successes.append(item)
                seen_success_seeds.add(seed)

                # 每成功一个立刻保存
                save_successes(successes)

            append_attempt_log(
                attempt_index=attempt_index,
                seed=seed,
                cube_pos=cube_pos,
                result=result,
                num_success_so_far=len(successes),
            )

            elapsed = time.time() - start_time
            success_added = len(successes) - success_before
            attempts_done = attempt_index

            rate = success_added / max(1, attempts_done)
            eta = None

            remaining_success = max(0, TARGET_SUCCESS - len(successes))
            if rate > 1e-6:
                eta = remaining_success / rate

            mark = "O" if place_success else ("P" if pick_success else "X")

            eta_text = "unknown"
            if eta is not None:
                eta_text = f"{eta / 60:.1f} min"

            print(
                f"{mark} attempt={attempt_index:04d}/{MAX_ATTEMPTS} "
                f"seed={seed} "
                f"cube=({cube_pos[0]:.3f},{cube_pos[1]:.3f}) "
                f"pick={pick_success} "
                f"place={place_success} "
                f"lift={float(result['lift_delta']):.4f} "
                f"xy={float(result['xy_dist']):.4f} "
                f"z_margin={float(result['z_margin']):.4f} "
                f"successes={len(successes)}/{TARGET_SUCCESS} "
                f"eta={eta_text}"
            )

    except KeyboardInterrupt:
        print("")
        print("收到 Ctrl+C，中断采集。正在保存已有成功样本...")

    save_successes(successes)

    elapsed = time.time() - start_time

    print("")
    print("=" * 80)
    print("Right-arm success trial collection 总结")
    print("=" * 80)
    print(f"本次运行 attempts：{attempt_index}")
    print(f"成功样本总数：{len(successes)}")
    print(f"本次新增成功：{len(successes) - success_before}")
    print(f"耗时：{elapsed / 60:.1f} min")
    print("保存文件：", OUTPUT_NPZ)
    print("日志文件：", OUTPUT_CSV)

    if successes:
        recent = successes[-min(10, len(successes)):]
        print("")
        print("最近成功样本：")
        for s in recent:
            print(
                f"seed={s['seed']} "
                f"cube=({s['cube_pos'][0]:.3f},{s['cube_pos'][1]:.3f}) "
                f"lift={s['lift_delta']:.4f} "
                f"xy={s['xy_dist']:.4f} "
                f"z_margin={s['z_margin']:.4f}"
            )


if __name__ == "__main__":
    main()
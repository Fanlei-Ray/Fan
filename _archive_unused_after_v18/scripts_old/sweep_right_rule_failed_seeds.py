from pathlib import Path
import sys

import mujoco
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import right_rule_pick_place as rr


# 刚才随机测试失败的 seeds
FAILED_SEEDS = [
    30002,
    30005,
    30006,
    30008,
    30010,
    30013,
    30015,
    30017,
    30020,
    30024,
    30029,
]

# 用同一个采样范围复现 cube 初始位置
CUBE_X_RANGE = (0.49, 0.55)
CUBE_Y_RANGE = (0.02, 0.08)
CUBE_Z = 1.050

# 再用全部 30 个 seeds 检查是否误伤
ALL_SEEDS = [30001 + i for i in range(1, 31)]


def sample_cube_pos(seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(CUBE_X_RANGE[0], CUBE_X_RANGE[1])
    y = rng.uniform(CUBE_Y_RANGE[0], CUBE_Y_RANGE[1])
    return np.array([x, y, CUBE_Z], dtype=float)


def make_config(name, pre_offset, grasp_offset, joint_biases):
    return {
        "name": name,
        "site_type": "tcp",
        "pregrasp_offset": np.array(pre_offset, dtype=float),
        "grasp_offset": np.array(grasp_offset, dtype=float),
        "preplace_offset": np.array([0.0, 0.0, 0.140], dtype=float),
        "place_offset": np.array([0.0, 0.0, 0.080], dtype=float),
        "joint_biases": dict(joint_biases),
    }


def generate_configs():
    configs = []

    # 当前成功 fixed-point 的 best
    base_pre = np.array([-0.006, -0.020, 0.140], dtype=float)
    base_grasp = np.array([-0.006, -0.020, 0.055], dtype=float)

    # 失败点集中在 y=0.05~0.08，说明右臂抓取侧向偏置可能还不对。
    x_values = [-0.014, -0.010, -0.006, -0.002, +0.002, +0.006]
    y_values = [-0.040, -0.032, -0.026, -0.020, -0.014, -0.008, 0.000]
    z_values = [0.040, 0.048, 0.055, 0.062, 0.070]

    joint7_values = [-0.100, -0.080, -0.060, -0.040, -0.020, 0.000]

    for x in x_values:
        for y in y_values:
            for z in z_values:
                for j7 in joint7_values:
                    grasp = np.array([x, y, z], dtype=float)

                    # pregrasp 保持比 grasp 高 8.5cm
                    pre = grasp + np.array([0.0, 0.0, 0.085], dtype=float)

                    name = f"tcp_x{x:+.3f}_y{y:+.3f}_z{z:+.3f}_j7{j7:+.3f}"

                    configs.append(
                        make_config(
                            name=name,
                            pre_offset=pre,
                            grasp_offset=grasp,
                            joint_biases={"right_joint7_ctrl": j7},
                        )
                    )

    # 把原始 best 放到最前面作为 base
    base = make_config(
        name="base_best_tcp_x-0.006_y-0.020_z+0.055_j7-0.060",
        pre_offset=base_pre,
        grasp_offset=base_grasp,
        joint_biases={"right_joint7_ctrl": -0.060},
    )

    return [base] + configs


def run_seed(model, site_name, seed, config):
    cube_pos = sample_cube_pos(seed)
    rr.FIXED_CUBE_POS = cube_pos.copy()

    result = rr.run_trial(
        model=model,
        site_name=site_name,
        config=config,
        do_place=True,
        viewer=None,
        realtime=False,
        data=None,
    )

    return {
        "seed": seed,
        "cube_pos": cube_pos,
        "pick_success": bool(result["pick_success"]),
        "place_success": bool(result["place_success"]),
        "lift_delta": float(result["lift_delta"]),
        "final_lift_delta": float(result["final_lift_delta"]),
        "xy_dist": float(result["xy_dist"]),
        "z_margin": float(result["z_margin"]),
        "cube_final": result["cube_final"].copy(),
        "frame_final": result["frame_final"].copy(),
    }


def eval_config_on_seeds(model, site_name, seeds, config):
    records = []

    for seed in seeds:
        r = run_seed(model, site_name, seed, config)
        records.append(r)

    pick_count = sum(1 for r in records if r["pick_success"])
    place_count = sum(1 for r in records if r["place_success"])
    avg_lift = float(np.mean([r["lift_delta"] for r in records]))

    return {
        "config": config,
        "records": records,
        "pick_count": pick_count,
        "place_count": place_count,
        "avg_lift": avg_lift,
    }


def print_eval_summary(prefix, result, n):
    config = result["config"]

    print(
        f"{prefix} {config['name']:52s} "
        f"pick={result['pick_count']:02d}/{n} "
        f"place={result['place_count']:02d}/{n} "
        f"avg_lift={result['avg_lift']:.4f} "
        f"grasp={np.array2string(config['grasp_offset'], precision=3)} "
        f"j7={config['joint_biases'].get('right_joint7_ctrl', 0.0):+.3f}"
    )


def main():
    print("=" * 90)
    print("Sweep right rule expert on failed seeds")
    print("=" * 90)
    print("FAILED_SEEDS:", FAILED_SEEDS)
    print("ALL_SEEDS:", ALL_SEEDS[0], "to", ALL_SEEDS[-1])
    print("CUBE_X_RANGE:", CUBE_X_RANGE)
    print("CUBE_Y_RANGE:", CUBE_Y_RANGE)
    print("=" * 90)

    model = mujoco.MjModel.from_xml_path(str(rr.XML_PATH))
    site_name = rr.choose_right_site(model)

    print("使用右臂 site:", site_name)

    configs = generate_configs()

    print("config 数量:", len(configs))
    print("=" * 90)

    failed_seed_results = []

    for i, config in enumerate(configs, start=1):
        result = eval_config_on_seeds(
            model=model,
            site_name=site_name,
            seeds=FAILED_SEEDS,
            config=config,
        )

        failed_seed_results.append(result)

        # 不要每个都刷屏太多，只打印比较好的
        if (
            i == 1
            or result["place_count"] >= 6
            or result["pick_count"] >= 8
        ):
            print_eval_summary("FAIL-SWEEP", result, len(FAILED_SEEDS))

    failed_seed_results.sort(
        key=lambda r: (
            r["place_count"],
            r["pick_count"],
            r["avg_lift"],
        ),
        reverse=True,
    )

    print("")
    print("=" * 90)
    print("TOP configs on failed seeds")
    print("=" * 90)

    for r in failed_seed_results[:15]:
        print_eval_summary("TOP-FAIL", r, len(FAILED_SEEDS))

    print("")
    print("=" * 90)
    print("Validate top configs on all 30 seeds")
    print("=" * 90)

    top_configs = [r["config"] for r in failed_seed_results[:10]]

    all_seed_results = []

    for config in top_configs:
        result = eval_config_on_seeds(
            model=model,
            site_name=site_name,
            seeds=ALL_SEEDS,
            config=config,
        )

        all_seed_results.append(result)
        print_eval_summary("ALL-30", result, len(ALL_SEEDS))

    all_seed_results.sort(
        key=lambda r: (
            r["place_count"],
            r["pick_count"],
            r["avg_lift"],
        ),
        reverse=True,
    )

    best = all_seed_results[0]
    best_config = best["config"]

    print("")
    print("=" * 90)
    print("BEST ON ALL 30")
    print("=" * 90)
    print("name:", best_config["name"])
    print("pick:", f"{best['pick_count']}/{len(ALL_SEEDS)}")
    print("place:", f"{best['place_count']}/{len(ALL_SEEDS)}")
    print("avg_lift:", best["avg_lift"])
    print("pregrasp_offset:", best_config["pregrasp_offset"])
    print("grasp_offset:", best_config["grasp_offset"])
    print("joint_biases:", best_config["joint_biases"])

    print("")
    print("失败样本：")
    for r in best["records"]:
        if not r["place_success"]:
            print(
                f"seed={r['seed']} "
                f"cube=({r['cube_pos'][0]:.3f},{r['cube_pos'][1]:.3f}) "
                f"pick={r['pick_success']} "
                f"place={r['place_success']} "
                f"lift={r['lift_delta']:.4f} "
                f"xy={r['xy_dist']:.4f} "
                f"z_margin={r['z_margin']:.4f}"
            )

    print("")
    print("=" * 90)
    print("下一步")
    print("=" * 90)
    print("如果 BEST ON ALL 30 明显高于 19/30，就把这个 config 固化到 test_right_rule_random.py。")


if __name__ == "__main__":
    main()
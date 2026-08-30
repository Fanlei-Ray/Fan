from pathlib import Path
import sys

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import test_bc_v4_residual_100 as base


NUM_EPISODES = 100
SEED_START = 20001

# 这就是目前唯一有正收益的 residual
RESIDUAL_X_BIAS = -0.010
RESIDUAL_Y_BIAS = 0.000
RESIDUAL_GAIN = 0.35


def gate_never(x, y):
    return False


def gate_always(x, y):
    return True


def gate_y_pos_030(x, y):
    return y >= 0.030


def gate_y_pos_035(x, y):
    return y >= 0.035


def gate_y_pos_040(x, y):
    return y >= 0.040


def gate_x_high_550(x, y):
    return x >= 0.550


def gate_x_high_555(x, y):
    return x >= 0.555


def gate_x_high_560(x, y):
    return x >= 0.560


def gate_y_pos_030_or_x_high_555(x, y):
    return (y >= 0.030) or (x >= 0.555)


def gate_y_pos_035_or_x_high_555(x, y):
    return (y >= 0.035) or (x >= 0.555)


def gate_failure_band_v1(x, y):
    # 来自失败样本分布：
    # 1. y 正向 0.03~0.06 是主失败带
    # 2. x 很大也容易失败
    # 3. y 负边界也有少量失败
    return (y >= 0.030) or (x >= 0.555) or (y <= -0.045)


def gate_failure_band_v2(x, y):
    # 稍微收紧正 y，减少误伤
    return (y >= 0.035) or (x >= 0.555) or (y <= -0.050)


def gate_failure_band_v3(x, y):
    # 更保守：只处理最明显危险区
    return (y >= 0.040) or (x >= 0.560) or (y <= -0.050)


GATE_CONFIGS = [
    ("base_no_residual", gate_never),
    ("always_alt_x_m010_g35", gate_always),

    ("gate_y_pos_030", gate_y_pos_030),
    ("gate_y_pos_035", gate_y_pos_035),
    ("gate_y_pos_040", gate_y_pos_040),

    ("gate_x_high_550", gate_x_high_550),
    ("gate_x_high_555", gate_x_high_555),
    ("gate_x_high_560", gate_x_high_560),

    ("gate_y030_or_x555", gate_y_pos_030_or_x_high_555),
    ("gate_y035_or_x555", gate_y_pos_035_or_x_high_555),

    ("gate_failure_band_v1", gate_failure_band_v1),
    ("gate_failure_band_v2", gate_failure_band_v2),
    ("gate_failure_band_v3", gate_failure_band_v3),
]


def active_config(active):
    if active:
        return {
            "name": "active_alt_x_m010_g35",
            "x_bias": RESIDUAL_X_BIAS,
            "y_bias": RESIDUAL_Y_BIAS,
            "gain": RESIDUAL_GAIN,
        }

    return {
        "name": "inactive_base",
        "x_bias": 0.000,
        "y_bias": 0.000,
        "gain": 0.00,
    }


def run_episode_gated(
    env,
    seed,
    gate_fn,
    model,
    phase_names,
    obs_aug_mean,
    obs_aug_std,
    action_mean,
    action_std,
    device,
):
    obs, info = env.reset(seed=seed)

    init_cube = info["cube_pos"].copy()
    init_x = float(init_cube[0])
    init_y = float(init_cube[1])

    residual_active = bool(gate_fn(init_x, init_y))
    config = active_config(residual_active)

    controller = base.PhaseController(
        x_bias=config["x_bias"],
        y_bias=config["y_bias"],
    )
    controller.reset(info)

    initial_cube_z = float(init_cube[2])
    max_cube_z = initial_cube_z

    lifted_ever = False
    success_ever = False
    final_info = info

    for step in range(base.MAX_STEPS):
        phase = controller.phase

        bc_act = base.policy_action(
            model=model,
            obs=obs,
            phase=phase,
            phase_names=phase_names,
            obs_aug_mean=obs_aug_mean,
            obs_aug_std=obs_aug_std,
            action_mean=action_mean,
            action_std=action_std,
            device=device,
        )

        action = base.apply_grasp_residual(
            env=env,
            info=info,
            controller=controller,
            bc_action_value=bc_act,
            gain=config["gain"],
        )

        obs, reward, terminated, truncated, info = env.step(action)
        final_info = info

        controller.update(obs, info)

        lifted_ever = lifted_ever or bool(info["cube_lifted"])
        success_ever = success_ever or bool(info["is_success"])
        max_cube_z = max(max_cube_z, float(info["cube_pos"][2]))

        if controller.phase == "done" and controller.phase_steps >= 5:
            break

        if truncated:
            break

    final_cube = final_info["cube_pos"].copy()
    final_frame = final_info["frame_pos"].copy()

    final_xy_dist = float(np.linalg.norm(final_cube[:2] - final_frame[:2]))
    final_z_margin = float(final_cube[2] - final_frame[2])
    max_lift_delta = float(max_cube_z - initial_cube_z)

    success = bool(success_ever or final_info["is_success"])

    category = base.classify_episode(
        success=success,
        lifted_ever=lifted_ever,
        max_lift_delta=max_lift_delta,
        final_xy_dist=final_xy_dist,
        final_z_margin=final_z_margin,
        final_phase=controller.phase,
    )

    return {
        "seed": seed,
        "success": success,
        "category": category,
        "init_x": init_x,
        "init_y": init_y,
        "xy_dist": final_xy_dist,
        "z_margin": final_z_margin,
        "max_lift_delta": max_lift_delta,
        "lifted_ever": lifted_ever,
        "success_ever": success_ever,
        "final_phase": controller.phase,
        "residual_active": residual_active,
    }


def summarize(name, records):
    success_count = sum(1 for r in records if r["success"])
    lifted_count = sum(1 for r in records if r["lifted_ever"])
    active_count = sum(1 for r in records if r["residual_active"])

    cats = [r["category"] for r in records]

    return {
        "name": name,
        "success_count": success_count,
        "lifted_count": lifted_count,
        "active_count": active_count,
        "pick_lift_fail": cats.count("pick_lift_fail"),
        "place_xy_fail": cats.count("place_xy_fail"),
        "place_z_fail": cats.count("place_z_fail"),
        "timeout": cats.count("timeout_or_phase_stuck"),
        "other_fail": (
            len(records)
            - success_count
            - cats.count("pick_lift_fail")
            - cats.count("place_xy_fail")
            - cats.count("place_z_fail")
            - cats.count("timeout_or_phase_stuck")
        ),
    }


def compare_to_base(all_results):
    base_records = all_results["base_no_residual"]
    base_by_seed = {r["seed"]: r for r in base_records}

    print("")
    print("=" * 90)
    print("相对 base 的救回/误伤统计")
    print("=" * 90)

    comparisons = []

    for name, records in all_results.items():
        if name == "base_no_residual":
            continue

        rescued = []
        broken = []

        for r in records:
            b = base_by_seed[r["seed"]]

            if (not b["success"]) and r["success"]:
                rescued.append(r)

            if b["success"] and (not r["success"]):
                broken.append(r)

        net = len(rescued) - len(broken)

        comparisons.append(
            {
                "name": name,
                "rescued": rescued,
                "broken": broken,
                "net": net,
            }
        )

    comparisons.sort(key=lambda x: (x["net"], len(x["rescued"])), reverse=True)

    for c in comparisons:
        print("")
        print("-" * 90)
        print(c["name"])
        print(f"救回：{len(c['rescued'])}")
        print(f"误伤：{len(c['broken'])}")
        print(f"净提升：{c['net']}")

        if c["rescued"]:
            print("救回 seeds:")
            print(", ".join(str(r["seed"]) for r in c["rescued"]))

        if c["broken"]:
            print("误伤 seeds:")
            print(", ".join(str(r["seed"]) for r in c["broken"]))

    return comparisons


def save_csv(all_results):
    csv_path = base.ROOT / "bc_v4_gated_residual_100_compare.csv"

    keys = [
        "config",
        "seed",
        "success",
        "category",
        "init_x",
        "init_y",
        "xy_dist",
        "z_margin",
        "max_lift_delta",
        "lifted_ever",
        "success_ever",
        "final_phase",
        "residual_active",
    ]

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")

        for config_name, records in all_results.items():
            for r in records:
                row = {"config": config_name}
                row.update(r)
                f.write(",".join(str(row[k]) for k in keys) + "\n")

    print("")
    print("已保存 CSV:", csv_path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("使用 device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    (
        model,
        phase_names,
        obs_aug_mean,
        obs_aug_std,
        action_mean,
        action_std,
        env_config,
    ) = base.load_policy(device)

    env = base.make_env(env_config)

    seeds = [SEED_START + i for i in range(NUM_EPISODES)]

    print("")
    print("=" * 90)
    print("BC v4 gated residual 100-episode 对比")
    print("=" * 90)
    print("Residual: x_bias=-0.010, y_bias=0.000, gain=0.35")
    print("Seeds:", seeds[0], "到", seeds[-1])
    print("Config 数量:", len(GATE_CONFIGS))
    print("=" * 90)

    all_results = {}

    for name, gate_fn in GATE_CONFIGS:
        print("")
        print("=" * 90)
        print("开始测试:", name)
        print("=" * 90)

        records = []

        for i, seed in enumerate(seeds, start=1):
            r = run_episode_gated(
                env=env,
                seed=seed,
                gate_fn=gate_fn,
                model=model,
                phase_names=phase_names,
                obs_aug_mean=obs_aug_mean,
                obs_aug_std=obs_aug_std,
                action_mean=action_mean,
                action_std=action_std,
                device=device,
            )

            records.append(r)

            mark = "O" if r["success"] else "X"
            active_mark = "A" if r["residual_active"] else "-"

            print(
                f"{mark}{active_mark} {name:24s} "
                f"ep={i:03d} seed={seed} "
                f"init=({r['init_x']:.3f},{r['init_y']:.3f}) "
                f"cat={r['category']:20s} "
                f"lift={r['max_lift_delta']:.3f} "
                f"xy={r['xy_dist']:.3f} "
                f"phase={r['final_phase']}"
            )

        all_results[name] = records

        s = summarize(name, records)
        print("")
        print(f"{name} 小结:")
        print(
            f"success={s['success_count']:03d}/{NUM_EPISODES}, "
            f"lifted={s['lifted_count']:03d}/{NUM_EPISODES}, "
            f"active={s['active_count']:03d}/{NUM_EPISODES}, "
            f"pick_fail={s['pick_lift_fail']:02d}, "
            f"place_fail={s['place_xy_fail']:02d}, "
            f"other={s['other_fail']:02d}"
        )

    env.close()

    summaries = [summarize(name, records) for name, records in all_results.items()]
    summaries.sort(
        key=lambda s: (
            s["success_count"],
            s["lifted_count"],
            -s["active_count"],
        ),
        reverse=True,
    )

    print("")
    print("=" * 90)
    print("总排名")
    print("=" * 90)

    for s in summaries:
        print(
            f"{s['name']:24s} "
            f"success={s['success_count']:03d}/{NUM_EPISODES} "
            f"lifted={s['lifted_count']:03d}/{NUM_EPISODES} "
            f"active={s['active_count']:03d}/{NUM_EPISODES} "
            f"pick_fail={s['pick_lift_fail']:02d} "
            f"place_fail={s['place_xy_fail']:02d} "
            f"other={s['other_fail']:02d}"
        )

    comparisons = compare_to_base(all_results)
    save_csv(all_results)

    print("")
    print("=" * 90)
    print("建议")
    print("=" * 90)

    best_summary = summaries[0]
    best_compare = next(
        (c for c in comparisons if c["name"] == best_summary["name"]),
        None,
    )

    print("最佳成功率 config:", best_summary["name"])
    print(f"成功率: {best_summary['success_count']}/{NUM_EPISODES}")

    if best_compare is not None:
        print(
            f"相对 base：救回 {len(best_compare['rescued'])}, "
            f"误伤 {len(best_compare['broken'])}, "
            f"净提升 {best_compare['net']}"
        )


if __name__ == "__main__":
    main()
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from openarm_pickplace_env import OpenArmPickPlaceEnv
import right_pick_place as pp


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "bc_policy_phase_v4.pt"

NUM_EPISODES = 100
MAX_STEPS = 300
RENDER = False

CUBE_X_RANGE = (0.48, 0.57)
CUBE_Y_RANGE = (-0.06, 0.06)

SEED_START = 20001

TEST_CONFIGS = [
    {
        "name": "base_no_residual",
        "x_bias": 0.000,
        "y_bias": 0.000,
        "gain": 0.00,
    },
    {
        "name": "best_y_m015_g50",
        "x_bias": 0.000,
        "y_bias": -0.015,
        "gain": 0.50,
    },
    {
        "name": "safer_y_m015_g35",
        "x_bias": 0.000,
        "y_bias": -0.015,
        "gain": 0.35,
    },
    {
        "name": "alt_x_m010_g35",
        "x_bias": -0.010,
        "y_bias": 0.000,
        "gain": 0.35,
    },
]


class PhaseBCPolicy(nn.Module):
    def __init__(self, input_dim, act_dim=5, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, x):
        return self.net(x)


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def phase_onehot(phase_name, phase_names):
    onehot = np.zeros((len(phase_names),), dtype=np.float32)

    if phase_name not in phase_names:
        raise ValueError(f"未知 phase: {phase_name}")

    onehot[phase_names.index(phase_name)] = 1.0
    return onehot


class PhaseController:
    def __init__(self, x_bias=0.0, y_bias=0.0):
        self.phase = "open_gripper"
        self.phase_steps = 0
        self.pick_cube_pos = None
        self.x_bias = float(x_bias)
        self.y_bias = float(y_bias)

    def reset(self, info):
        self.phase = "open_gripper"
        self.phase_steps = 0
        self.pick_cube_pos = info["cube_pos"].copy()

    def _switch(self, next_phase):
        self.phase = next_phase
        self.phase_steps = 0

    def grasp_bias(self):
        return np.array([self.x_bias, self.y_bias, 0.0], dtype=np.float32)

    def update(self, obs, info):
        self.phase_steps += 1

        tcp_pos = info["tcp_pos"]
        frame_pos = info["frame_pos"]

        finger_ctrl = float(obs[22])
        lifter_ctrl = float(obs[23])

        if self.pick_cube_pos is None:
            self.pick_cube_pos = info["cube_pos"].copy()

        if self.phase == "open_gripper":
            if finger_ctrl >= pp.LEFT_FINGER_PRE_OPEN - 0.02 or self.phase_steps > 18:
                self.pick_cube_pos = info["cube_pos"].copy()
                self._switch("move_pregrasp")
            return

        if self.phase == "move_pregrasp":
            target = self.pick_cube_pos + pp.PICK_PREGRASP_OFFSET + self.grasp_bias()

            if np.linalg.norm(tcp_pos - target) < 0.030 or self.phase_steps > 85:
                self._switch("move_grasp")
            return

        if self.phase == "move_grasp":
            target = self.pick_cube_pos + pp.PICK_GRASP_OFFSET + self.grasp_bias()

            if np.linalg.norm(tcp_pos - target) < 0.025 or self.phase_steps > 95:
                self._switch("close_gripper")
            return

        if self.phase == "close_gripper":
            if finger_ctrl <= pp.LEFT_FINGER_CLOSE + 0.04 or self.phase_steps > 40:
                self._switch("hold_grasp")
            return

        if self.phase == "hold_grasp":
            if self.phase_steps > 10:
                self._switch("lift_object")
            return

        if self.phase == "lift_object":
            if info["cube_lifted"] or lifter_ctrl >= pp.LIFTER_UP - 0.015 or self.phase_steps > 50:
                self._switch("move_preplace")
            return

        if self.phase == "move_preplace":
            target = frame_pos + pp.PLACE_PREPLACE_OFFSET

            if info["is_success"] or np.linalg.norm(tcp_pos - target) < 0.050 or self.phase_steps > 120:
                self._switch("move_release")
            return

        if self.phase == "move_release":
            target = frame_pos + pp.PLACE_RELEASE_OFFSET

            if np.linalg.norm(tcp_pos - target) < 0.040 or self.phase_steps > 90:
                self._switch("open_release")
            return

        if self.phase == "open_release":
            if finger_ctrl >= pp.LEFT_FINGER_OPEN - 0.04 or self.phase_steps > 35:
                self._switch("done")
            return

        if self.phase == "done":
            return


def load_policy(device):
    checkpoint = safe_torch_load(MODEL_PATH, device)

    model = PhaseBCPolicy(
        input_dim=int(checkpoint["input_dim"]),
        act_dim=int(checkpoint["act_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    phase_names = list(checkpoint["phase_names"])

    obs_aug_mean = checkpoint["obs_aug_mean"].to(device)
    obs_aug_std = checkpoint["obs_aug_std"].to(device)

    action_mean = checkpoint["action_mean"].to(device)
    action_std = checkpoint["action_std"].to(device)

    env_config = checkpoint.get("env_config", {})

    print("已加载模型:", MODEL_PATH)
    print("best_epoch:", checkpoint.get("epoch"))
    print("best_val_loss:", checkpoint.get("best_val_loss"))
    print("val_mae:", checkpoint.get("val_mae"))
    print("env_config:", env_config)

    return (
        model,
        phase_names,
        obs_aug_mean,
        obs_aug_std,
        action_mean,
        action_std,
        env_config,
    )


def policy_action(
    model,
    obs,
    phase,
    phase_names,
    obs_aug_mean,
    obs_aug_std,
    action_mean,
    action_std,
    device,
):
    p = phase_onehot(phase, phase_names)
    obs_aug = np.concatenate([obs.astype(np.float32), p], axis=0).astype(np.float32)

    obs_tensor = torch.from_numpy(obs_aug).float().to(device)
    obs_norm = (obs_tensor - obs_aug_mean) / obs_aug_std

    with torch.no_grad():
        action_norm = model(obs_norm.unsqueeze(0)).squeeze(0)
        action = action_norm * action_std + action_mean

    return np.clip(action.cpu().numpy().astype(np.float32), -1.0, 1.0)


def make_env(env_config):
    env = OpenArmPickPlaceEnv(
        render_mode="human" if RENDER else None,
        randomize_cube=True,
        cube_x_range=CUBE_X_RANGE,
        cube_y_range=CUBE_Y_RANGE,
        max_episode_steps=MAX_STEPS,
    )

    if "frame_skip" in env_config:
        env.frame_skip = int(env_config["frame_skip"])

    if "max_tcp_delta" in env_config:
        env.max_tcp_delta = float(env_config["max_tcp_delta"])

    if "max_finger_delta" in env_config:
        env.max_finger_delta = float(env_config["max_finger_delta"])

    if "max_lifter_delta" in env_config:
        env.max_lifter_delta = float(env_config["max_lifter_delta"])

    return env


def tcp_rule_action(env, info, target):
    tcp_pos = info["tcp_pos"].astype(np.float32)
    delta = np.asarray(target, dtype=np.float32) - tcp_pos
    return np.clip(delta / env.max_tcp_delta, -1.0, 1.0).astype(np.float32)


def apply_grasp_residual(env, info, controller, bc_action_value, gain):
    if gain <= 0.0:
        return bc_action_value.astype(np.float32)

    if controller.pick_cube_pos is None:
        return bc_action_value.astype(np.float32)

    # 只修抓取接近阶段，放置阶段完全交还给 BC
    if controller.phase not in ["move_pregrasp", "move_grasp", "close_gripper"]:
        return bc_action_value.astype(np.float32)

    if controller.phase == "move_pregrasp":
        target = controller.pick_cube_pos + pp.PICK_PREGRASP_OFFSET + controller.grasp_bias()
    else:
        target = controller.pick_cube_pos + pp.PICK_GRASP_OFFSET + controller.grasp_bias()

    rule_tcp = tcp_rule_action(env, info, target)

    action = bc_action_value.copy()
    action[:3] = np.clip(
        (1.0 - gain) * bc_action_value[:3] + gain * rule_tcp,
        -1.0,
        1.0,
    )

    return action.astype(np.float32)


def classify_episode(success, lifted_ever, max_lift_delta, final_xy_dist, final_z_margin, final_phase):
    if success:
        return "success"

    if not lifted_ever and max_lift_delta < 0.015:
        return "pick_lift_fail"

    if lifted_ever and final_xy_dist > 0.045:
        return "place_xy_fail"

    if lifted_ever and final_z_margin <= 0.015:
        return "place_z_fail"

    if final_phase != "done":
        return "timeout_or_phase_stuck"

    return "other_fail"


def run_episode(
    env,
    seed,
    config,
    model,
    phase_names,
    obs_aug_mean,
    obs_aug_std,
    action_mean,
    action_std,
    device,
):
    obs, info = env.reset(seed=seed)

    controller = PhaseController(
        x_bias=config["x_bias"],
        y_bias=config["y_bias"],
    )
    controller.reset(info)

    init_cube = info["cube_pos"].copy()
    initial_cube_z = float(init_cube[2])
    max_cube_z = initial_cube_z

    lifted_ever = False
    success_ever = False
    final_info = info

    for step in range(MAX_STEPS):
        phase = controller.phase

        bc_act = policy_action(
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

        action = apply_grasp_residual(
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

    category = classify_episode(
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
        "init_x": float(init_cube[0]),
        "init_y": float(init_cube[1]),
        "final_x": float(final_cube[0]),
        "final_y": float(final_cube[1]),
        "final_z": float(final_cube[2]),
        "frame_x": float(final_frame[0]),
        "frame_y": float(final_frame[1]),
        "frame_z": float(final_frame[2]),
        "xy_dist": final_xy_dist,
        "z_margin": final_z_margin,
        "max_lift_delta": max_lift_delta,
        "lifted_ever": lifted_ever,
        "success_ever": success_ever,
        "final_phase": controller.phase,
    }


def summarize_records(name, records):
    success_count = sum(1 for r in records if r["success"])
    lifted_count = sum(1 for r in records if r["lifted_ever"])

    cats = [r["category"] for r in records]

    print("")
    print("=" * 80)
    print(f"{name} 总结")
    print("=" * 80)
    print(f"成功次数：{success_count}/{len(records)}")
    print(f"成功率：{success_count / len(records) * 100:.1f}%")
    print(f"lifted_ever：{lifted_count}/{len(records)}")
    print(f"lifted_ever 比例：{lifted_count / len(records) * 100:.1f}%")

    print("")
    print("类别统计：")
    for c in sorted(set(cats)):
        print(f"{c}: {cats.count(c)}")

    print("")
    print("失败样本：")
    for r in records:
        if not r["success"]:
            print(
                f"seed={r['seed']} "
                f"init=({r['init_x']:.3f},{r['init_y']:.3f}) "
                f"cat={r['category']} "
                f"xy={r['xy_dist']:.3f} "
                f"z_margin={r['z_margin']:.3f} "
                f"lift={r['max_lift_delta']:.3f} "
                f"phase={r['final_phase']}"
            )

    return success_count


def save_csv(all_results):
    csv_path = ROOT / "bc_v4_residual_100_compare.csv"

    keys = [
        "config",
        "seed",
        "success",
        "category",
        "init_x",
        "init_y",
        "final_x",
        "final_y",
        "final_z",
        "frame_x",
        "frame_y",
        "frame_z",
        "xy_dist",
        "z_margin",
        "max_lift_delta",
        "lifted_ever",
        "success_ever",
        "final_phase",
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


def compare_to_base(all_results):
    base = all_results["base_no_residual"]

    base_by_seed = {r["seed"]: r for r in base}

    print("")
    print("=" * 80)
    print("相对 base 的救回/误伤统计")
    print("=" * 80)

    for config in TEST_CONFIGS:
        name = config["name"]

        if name == "base_no_residual":
            continue

        records = all_results[name]

        rescued = []
        broken = []

        for r in records:
            b = base_by_seed[r["seed"]]

            if (not b["success"]) and r["success"]:
                rescued.append(r)

            if b["success"] and (not r["success"]):
                broken.append(r)

        print("")
        print("-" * 80)
        print(name)
        print(f"救回：{len(rescued)}")
        print(f"误伤：{len(broken)}")
        print(f"净提升：{len(rescued) - len(broken)}")

        if rescued:
            print("救回 seeds:")
            print(", ".join(str(r["seed"]) for r in rescued))

        if broken:
            print("误伤 seeds:")
            print(", ".join(str(r["seed"]) for r in broken))


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
    ) = load_policy(device)

    env = make_env(env_config)

    seeds = [SEED_START + i for i in range(NUM_EPISODES)]

    print("")
    print("=" * 80)
    print("BC v4 residual 100-episode 正式对比")
    print("=" * 80)
    print("模型:", MODEL_PATH)
    print("NUM_EPISODES:", NUM_EPISODES)
    print("MAX_STEPS:", MAX_STEPS)
    print("CUBE_X_RANGE:", CUBE_X_RANGE)
    print("CUBE_Y_RANGE:", CUBE_Y_RANGE)
    print("seeds:", seeds[0], "到", seeds[-1])
    print("=" * 80)

    all_results = {}

    for config in TEST_CONFIGS:
        name = config["name"]

        print("")
        print("=" * 80)
        print(
            f"开始测试 config={name}, "
            f"x_bias={config['x_bias']:+.3f}, "
            f"y_bias={config['y_bias']:+.3f}, "
            f"gain={config['gain']:.2f}"
        )
        print("=" * 80)

        records = []

        for i, seed in enumerate(seeds, start=1):
            r = run_episode(
                env=env,
                seed=seed,
                config=config,
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
            print(
                f"{mark} {name:18s} "
                f"ep={i:03d} seed={seed} "
                f"init=({r['init_x']:.3f},{r['init_y']:.3f}) "
                f"cat={r['category']:20s} "
                f"lift={r['max_lift_delta']:.3f} "
                f"xy={r['xy_dist']:.3f} "
                f"phase={r['final_phase']}"
            )

        all_results[name] = records
        summarize_records(name, records)

    env.close()

    print("")
    print("=" * 80)
    print("总排名")
    print("=" * 80)

    ranking = []

    for name, records in all_results.items():
        success_count = sum(1 for r in records if r["success"])
        lifted_count = sum(1 for r in records if r["lifted_ever"])
        cats = [r["category"] for r in records]

        ranking.append(
            {
                "name": name,
                "success_count": success_count,
                "lifted_count": lifted_count,
                "pick_lift_fail": cats.count("pick_lift_fail"),
                "place_xy_fail": cats.count("place_xy_fail"),
                "other_fail": len(records) - success_count - cats.count("pick_lift_fail") - cats.count("place_xy_fail"),
            }
        )

    ranking.sort(key=lambda x: (x["success_count"], x["lifted_count"]), reverse=True)

    for r in ranking:
        print(
            f"{r['name']:18s} "
            f"success={r['success_count']:03d}/{NUM_EPISODES} "
            f"lifted={r['lifted_count']:03d}/{NUM_EPISODES} "
            f"pick_fail={r['pick_lift_fail']:02d} "
            f"place_fail={r['place_xy_fail']:02d} "
            f"other={r['other_fail']:02d}"
        )

    compare_to_base(all_results)
    save_csv(all_results)


if __name__ == "__main__":
    main()
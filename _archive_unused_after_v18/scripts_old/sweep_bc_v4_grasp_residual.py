from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from openarm_pickplace_env import OpenArmPickPlaceEnv
import right_pick_place as pp


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "bc_policy_phase_v4.pt"

MAX_STEPS = 300
RENDER = False

CUBE_X_RANGE = (0.48, 0.57)
CUBE_Y_RANGE = (-0.06, 0.06)

# 这是刚才 100 次测试里面失败的 seed
TARGET_SEEDS = [
    20001,
    20004,
    20015,
    20024,
    20036,
    20041,
    20048,
    20053,
    20054,
    20058,
    20060,
    20071,
    20073,
    20077,
    20078,
    20079,
    20081,
    20088,
    20092,
    20094,
    20099,
]

# 扫一组抓取 TCP residual。
# x_bias / y_bias 单位是米。
# gain 是把 BC action 和规则修正 action 混合的比例。
SWEEP_CONFIGS = [
    ("base_no_residual", 0.000, 0.000, 0.00),

    ("y_m020_g35", 0.000, -0.020, 0.35),
    ("y_m015_g35", 0.000, -0.015, 0.35),
    ("y_m010_g35", 0.000, -0.010, 0.35),
    ("y_m005_g35", 0.000, -0.005, 0.35),

    ("y_p005_g35", 0.000, 0.005, 0.35),
    ("y_p010_g35", 0.000, 0.010, 0.35),
    ("y_p015_g35", 0.000, 0.015, 0.35),
    ("y_p020_g35", 0.000, 0.020, 0.35),

    ("y_m015_g50", 0.000, -0.015, 0.50),
    ("y_m010_g50", 0.000, -0.010, 0.50),
    ("y_p010_g50", 0.000, 0.010, 0.50),
    ("y_p015_g50", 0.000, 0.015, 0.50),

    ("x_m010_g35", -0.010, 0.000, 0.35),
    ("x_p010_g35", 0.010, 0.000, 0.35),

    ("x_m008_y_m010_g35", -0.008, -0.010, 0.35),
    ("x_m008_y_p010_g35", -0.008, 0.010, 0.35),
    ("x_p008_y_m010_g35", 0.008, -0.010, 0.35),
    ("x_p008_y_p010_g35", 0.008, 0.010, 0.35),
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

    return model, phase_names, obs_aug_mean, obs_aug_std, action_mean, action_std, env_config


def policy_action(model, obs, phase, phase_names, obs_aug_mean, obs_aug_std, action_mean, action_std, device):
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


def apply_grasp_residual(env, obs, info, controller, bc_act, gain):
    if gain <= 0.0:
        return bc_act

    if controller.pick_cube_pos is None:
        return bc_act

    # 只在接近和闭合阶段修正 TCP，其他阶段完全保留 BC
    if controller.phase not in ["move_pregrasp", "move_grasp", "close_gripper"]:
        return bc_act

    if controller.phase == "move_pregrasp":
        target = controller.pick_cube_pos + pp.PICK_PREGRASP_OFFSET + controller.grasp_bias()
    else:
        target = controller.pick_cube_pos + pp.PICK_GRASP_OFFSET + controller.grasp_bias()

    rule_tcp = tcp_rule_action(env, info, target)

    act = bc_act.copy()
    act[:3] = np.clip((1.0 - gain) * bc_act[:3] + gain * rule_tcp, -1.0, 1.0)

    return act.astype(np.float32)


def run_one_episode(
    env,
    seed,
    x_bias,
    y_bias,
    gain,
    model,
    phase_names,
    obs_aug_mean,
    obs_aug_std,
    action_mean,
    action_std,
    device,
):
    obs, info = env.reset(seed=seed)

    controller = PhaseController(x_bias=x_bias, y_bias=y_bias)
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
            model,
            obs,
            phase,
            phase_names,
            obs_aug_mean,
            obs_aug_std,
            action_mean,
            action_std,
            device,
        )

        action = apply_grasp_residual(
            env=env,
            obs=obs,
            info=info,
            controller=controller,
            bc_act=bc_act,
            gain=gain,
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

    xy_dist = float(np.linalg.norm(final_cube[:2] - final_frame[:2]))
    z_margin = float(final_cube[2] - final_frame[2])
    max_lift_delta = float(max_cube_z - initial_cube_z)

    success = bool(success_ever or final_info["is_success"])

    if success:
        category = "success"
    elif not lifted_ever and max_lift_delta < 0.015:
        category = "pick_lift_fail"
    elif lifted_ever and xy_dist > 0.045:
        category = "place_xy_fail"
    elif success_ever and not success:
        category = "unstable_after_success"
    elif controller.phase != "done":
        category = "timeout_or_phase_stuck"
    else:
        category = "other_fail"

    return {
        "seed": seed,
        "success": success,
        "lifted_ever": lifted_ever,
        "success_ever": success_ever,
        "category": category,
        "init_x": float(init_cube[0]),
        "init_y": float(init_cube[1]),
        "xy_dist": xy_dist,
        "z_margin": z_margin,
        "max_lift_delta": max_lift_delta,
        "final_phase": controller.phase,
    }


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

    print("")
    print("=" * 90)
    print("BC v4 grasp residual sweep")
    print("=" * 90)
    print("失败 seeds 数量:", len(TARGET_SEEDS))
    print("MAX_STEPS:", MAX_STEPS)
    print("范围:", CUBE_X_RANGE, CUBE_Y_RANGE)
    print("=" * 90)

    all_summaries = []

    for name, x_bias, y_bias, gain in SWEEP_CONFIGS:
        records = []

        for seed in TARGET_SEEDS:
            r = run_one_episode(
                env=env,
                seed=seed,
                x_bias=x_bias,
                y_bias=y_bias,
                gain=gain,
                model=model,
                phase_names=phase_names,
                obs_aug_mean=obs_aug_mean,
                obs_aug_std=obs_aug_std,
                action_mean=action_mean,
                action_std=action_std,
                device=device,
            )
            records.append(r)

        success_count = sum(1 for r in records if r["success"])
        lifted_count = sum(1 for r in records if r["lifted_ever"])

        cats = [r["category"] for r in records]
        pick_fail_count = cats.count("pick_lift_fail")
        place_fail_count = cats.count("place_xy_fail")
        stuck_count = cats.count("timeout_or_phase_stuck")
        other_fail_count = len(records) - success_count - pick_fail_count - place_fail_count - stuck_count

        summary = {
            "name": name,
            "x_bias": x_bias,
            "y_bias": y_bias,
            "gain": gain,
            "success_count": success_count,
            "lifted_count": lifted_count,
            "pick_lift_fail": pick_fail_count,
            "place_xy_fail": place_fail_count,
            "stuck": stuck_count,
            "other_fail": other_fail_count,
            "records": records,
        }

        all_summaries.append(summary)

        print(
            f"{name:22s} "
            f"x_bias={x_bias:+.3f} "
            f"y_bias={y_bias:+.3f} "
            f"gain={gain:.2f} | "
            f"success={success_count:02d}/{len(TARGET_SEEDS)} "
            f"lifted={lifted_count:02d}/{len(TARGET_SEEDS)} "
            f"pick_fail={pick_fail_count:02d} "
            f"place_fail={place_fail_count:02d} "
            f"stuck={stuck_count:02d} "
            f"other={other_fail_count:02d}"
        )

    env.close()

    all_summaries.sort(
        key=lambda s: (
            s["success_count"],
            s["lifted_count"],
            -s["pick_lift_fail"],
        ),
        reverse=True,
    )

    print("")
    print("=" * 90)
    print("TOP configs")
    print("=" * 90)

    for s in all_summaries[:8]:
        print(
            f"{s['name']:22s} "
            f"x_bias={s['x_bias']:+.3f} "
            f"y_bias={s['y_bias']:+.3f} "
            f"gain={s['gain']:.2f} | "
            f"success={s['success_count']:02d}/{len(TARGET_SEEDS)} "
            f"lifted={s['lifted_count']:02d}/{len(TARGET_SEEDS)} "
            f"pick_fail={s['pick_lift_fail']:02d} "
            f"place_fail={s['place_xy_fail']:02d} "
            f"stuck={s['stuck']:02d} "
            f"other={s['other_fail']:02d}"
        )

    best = all_summaries[0]

    print("")
    print("=" * 90)
    print("BEST config 失败样本")
    print("=" * 90)
    print(
        f"BEST: {best['name']} "
        f"x_bias={best['x_bias']:+.3f}, "
        f"y_bias={best['y_bias']:+.3f}, "
        f"gain={best['gain']:.2f}"
    )

    for r in best["records"]:
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

    print("")
    print("建议下一步：如果某个 config 明显比 base 好，再把它放进 100 episode 正式测试。")


if __name__ == "__main__":
    main()
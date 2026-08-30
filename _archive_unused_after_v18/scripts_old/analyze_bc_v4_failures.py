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
    onehot[phase_names.index(phase_name)] = 1.0
    return onehot


class PhaseController:
    def __init__(self):
        self.phase = "open_gripper"
        self.phase_steps = 0
        self.pick_cube_pos = None

    def reset(self, info):
        self.phase = "open_gripper"
        self.phase_steps = 0
        self.pick_cube_pos = info["cube_pos"].copy()

    def _switch(self, next_phase):
        self.phase = next_phase
        self.phase_steps = 0

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
            target = self.pick_cube_pos + pp.PICK_PREGRASP_OFFSET
            if np.linalg.norm(tcp_pos - target) < 0.030 or self.phase_steps > 85:
                self._switch("move_grasp")
            return

        if self.phase == "move_grasp":
            target = self.pick_cube_pos + pp.PICK_GRASP_OFFSET
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

    return (
        model,
        list(checkpoint["phase_names"]),
        checkpoint["obs_aug_mean"].to(device),
        checkpoint["obs_aug_std"].to(device),
        checkpoint["action_mean"].to(device),
        checkpoint["action_std"].to(device),
        checkpoint.get("env_config", {}),
    )


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


def classify_episode(success, lifted_ever, success_ever, max_lift_delta, final_xy_dist, final_z_margin, final_phase):
    if success:
        return "success"

    if not lifted_ever and max_lift_delta < 0.015:
        return "pick_lift_fail"

    if lifted_ever and final_xy_dist > 0.045:
        return "place_xy_fail"

    if lifted_ever and final_z_margin <= 0.015:
        return "place_z_fail"

    if success_ever and not success:
        return "unstable_after_success"

    if final_phase not in ["done"]:
        return "timeout_or_phase_stuck"

    return "other_fail"


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

    print("模型:", MODEL_PATH)
    print("env_config:", env_config)
    print("测试 episodes:", NUM_EPISODES)
    print("范围:", CUBE_X_RANGE, CUBE_Y_RANGE)

    env = make_env(env_config)

    records = []

    for ep in range(1, NUM_EPISODES + 1):
        seed = 20000 + ep
        obs, info = env.reset(seed=seed)

        controller = PhaseController()
        controller.reset(info)

        init_cube = info["cube_pos"].copy()
        initial_cube_z = float(init_cube[2])
        max_cube_z = initial_cube_z

        lifted_ever = False
        success_ever = False
        success = False
        final_info = info

        for step in range(MAX_STEPS):
            phase = controller.phase

            action = policy_action(
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

            obs, reward, terminated, truncated, info = env.step(action)
            final_info = info

            controller.update(obs, info)

            lifted_ever = lifted_ever or bool(info["cube_lifted"])
            success_ever = success_ever or bool(info["is_success"])
            max_cube_z = max(max_cube_z, float(info["cube_pos"][2]))

            if controller.phase == "done" and controller.phase_steps >= 5:
                success = bool(success_ever or info["is_success"])
                break

            if truncated:
                success = bool(success_ever or info["is_success"])
                break

        final_cube = final_info["cube_pos"].copy()
        final_frame = final_info["frame_pos"].copy()

        final_xy_dist = float(np.linalg.norm(final_cube[:2] - final_frame[:2]))
        final_z_margin = float(final_cube[2] - final_frame[2])
        max_lift_delta = float(max_cube_z - initial_cube_z)

        category = classify_episode(
            success=success,
            lifted_ever=lifted_ever,
            success_ever=success_ever,
            max_lift_delta=max_lift_delta,
            final_xy_dist=final_xy_dist,
            final_z_margin=final_z_margin,
            final_phase=controller.phase,
        )

        record = {
            "ep": ep,
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
            "xy_dist": final_xy_dist,
            "z_margin": final_z_margin,
            "max_lift_delta": max_lift_delta,
            "lifted_ever": lifted_ever,
            "success_ever": success_ever,
            "final_phase": controller.phase,
        }

        records.append(record)

        mark = "O" if success else "X"
        print(
            f"{mark} ep={ep:03d} seed={seed} "
            f"init=({record['init_x']:.3f},{record['init_y']:.3f}) "
            f"cat={category:22s} "
            f"xy={final_xy_dist:.3f} "
            f"lift={max_lift_delta:.3f} "
            f"phase={controller.phase}"
        )

    env.close()

    success_count = sum(1 for r in records if r["success"])

    print("")
    print("=" * 80)
    print("BC v4 failure analysis 总结")
    print("=" * 80)
    print(f"成功次数：{success_count}/{NUM_EPISODES}")
    print(f"成功率：{success_count / NUM_EPISODES * 100:.1f}%")

    print("")
    print("失败/成功类别统计：")
    cats = [r["category"] for r in records]
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

    # 保存 CSV
    csv_path = ROOT / "bc_v4_failure_analysis.csv"
    keys = list(records[0].keys())

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in records:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

    print("")
    print("已保存 CSV:", csv_path)


if __name__ == "__main__":
    main()
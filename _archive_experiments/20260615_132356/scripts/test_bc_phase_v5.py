from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from openarm_pickplace_env import OpenArmPickPlaceEnv
import right_pick_place as pp


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "bc_policy_phase_v5_1.pt"

NUM_EPISODES = 30
MAX_STEPS = 300
RENDER = False

TEST_CONFIGS = [
    {
        "name": "v5_1_wide",
        "cube_x_range": (0.48, 0.57),
        "cube_y_range": (-0.06, 0.06),
    },
    {
        "name": "v5_1_harder",
        "cube_x_range": (0.47, 0.58),
        "cube_y_range": (-0.08, 0.08),
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
            if finger_ctrl >= pp.LEFT_FINGER_PRE_OPEN - 0.02 or self.phase_steps > 22:
                self.pick_cube_pos = info["cube_pos"].copy()
                self._switch("move_pregrasp")
            return

        if self.phase == "move_pregrasp":
            target = self.pick_cube_pos + pp.PICK_PREGRASP_OFFSET
            if np.linalg.norm(tcp_pos - target) < 0.035 or self.phase_steps > 95:
                self._switch("move_grasp")
            return

        if self.phase == "move_grasp":
            target = self.pick_cube_pos + pp.PICK_GRASP_OFFSET
            if np.linalg.norm(tcp_pos - target) < 0.030 or self.phase_steps > 105:
                self._switch("close_gripper")
            return

        if self.phase == "close_gripper":
            if finger_ctrl <= pp.LEFT_FINGER_CLOSE + 0.04 or self.phase_steps > 48:
                self._switch("hold_grasp")
            return

        if self.phase == "hold_grasp":
            if self.phase_steps > 12:
                self._switch("lift_object")
            return

        if self.phase == "lift_object":
            if info["cube_lifted"] or lifter_ctrl >= pp.LIFTER_UP - 0.015 or self.phase_steps > 60:
                self._switch("move_preplace")
            return

        if self.phase == "move_preplace":
            target = frame_pos + pp.PLACE_PREPLACE_OFFSET
            if info["is_success"] or np.linalg.norm(tcp_pos - target) < 0.055 or self.phase_steps > 135:
                self._switch("move_release")
            return

        if self.phase == "move_release":
            target = frame_pos + pp.PLACE_RELEASE_OFFSET
            if np.linalg.norm(tcp_pos - target) < 0.045 or self.phase_steps > 105:
                self._switch("open_release")
            return

        if self.phase == "open_release":
            if finger_ctrl >= pp.LEFT_FINGER_OPEN - 0.04 or self.phase_steps > 42:
                self._switch("done")
            return

        if self.phase == "done":
            return


def load_policy(path, device):
    checkpoint = safe_torch_load(path, device)

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

    print("已加载模型:", path)
    print("best_epoch:", checkpoint.get("epoch"))
    print("best_val_loss:", checkpoint.get("best_val_loss"))
    print("val_mae:", checkpoint.get("val_mae"))
    print("phase_names:", phase_names)
    print("env_config:", env_config)

    return model, phase_names, obs_aug_mean, obs_aug_std, action_mean, action_std, env_config


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

    action = action.cpu().numpy().astype(np.float32)
    action = np.clip(action, -1.0, 1.0)

    return action


def make_env(config, env_config):
    env = OpenArmPickPlaceEnv(
        render_mode="human" if RENDER else None,
        randomize_cube=True,
        cube_x_range=config["cube_x_range"],
        cube_y_range=config["cube_y_range"],
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


def run_one_config(
    config,
    model,
    phase_names,
    obs_aug_mean,
    obs_aug_std,
    action_mean,
    action_std,
    env_config,
    device,
):
    env = make_env(config, env_config)

    success_count = 0
    episode_returns = []
    final_phases = []

    print("")
    print("=" * 60)
    print("开始测试:", config["name"])
    print("cube_x_range:", config["cube_x_range"])
    print("cube_y_range:", config["cube_y_range"])
    print("=" * 60)

    for ep in range(1, NUM_EPISODES + 1):
        obs, info = env.reset(seed=70000 + ep)
        controller = PhaseController()
        controller.reset(info)

        total_reward = 0.0
        success = False
        final_info = info

        for step in range(MAX_STEPS):
            phase = controller.phase

            action = policy_action(
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

            next_obs, reward, terminated, truncated, next_info = env.step(action)

            total_reward += float(reward)

            obs = next_obs
            info = next_info
            final_info = next_info

            controller.update(obs, info)

            if controller.phase == "done" and controller.phase_steps >= 5:
                success = bool(info["is_success"])
                break

            if truncated:
                success = bool(info["is_success"])
                break

        if success:
            success_count += 1

        episode_returns.append(total_reward)
        final_phases.append(controller.phase)

        print(
            f"Episode {ep:02d} | "
            f"success={success} | "
            f"phase={controller.phase:12s} | "
            f"return={total_reward:.3f} | "
            f"cube={np.array2string(final_info['cube_pos'], precision=3)} | "
            f"frame={np.array2string(final_info['frame_pos'], precision=3)}"
        )

    env.close()

    print("")
    print("=" * 60)
    print(f"BC phase v5 测试总结：{config['name']}")
    print("=" * 60)
    print(f"成功次数：{success_count}/{NUM_EPISODES}")
    print(f"成功率：{success_count / NUM_EPISODES * 100:.1f}%")
    print(f"平均 return：{np.mean(episode_returns):.3f}")

    print("")
    print("final phase 统计：")
    for phase in sorted(set(final_phases)):
        print(f"{phase}: {final_phases.count(phase)}")

    return success_count / NUM_EPISODES


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型：{MODEL_PATH}")

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
    ) = load_policy(MODEL_PATH, device)

    results = {}

    for config in TEST_CONFIGS:
        rate = run_one_config(
            config=config,
            model=model,
            phase_names=phase_names,
            obs_aug_mean=obs_aug_mean,
            obs_aug_std=obs_aug_std,
            action_mean=action_mean,
            action_std=action_std,
            env_config=env_config,
            device=device,
        )
        results[config["name"]] = rate

    print("")
    print("=" * 60)
    print("BC phase v5 全部测试完成")
    print("=" * 60)
    for name, rate in results.items():
        print(f"{name}: {rate * 100:.1f}%")


if __name__ == "__main__":
    main()
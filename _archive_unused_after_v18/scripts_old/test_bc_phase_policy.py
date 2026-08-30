from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from openarm_pickplace_env import OpenArmPickPlaceEnv
import right_pick_place as pp


ROOT = Path(__file__).resolve().parents[1]

MODEL_CANDIDATES = [
    ROOT / "bc_policy_phase_v4.pt",
]

NUM_EPISODES = 30
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


def choose_model_path():
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "找不到 BC policy 模型。请确认至少存在以下文件之一：\n"
        + "\n".join(str(p) for p in MODEL_CANDIDATES)
    )


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
    """
    原 v4 controller 恢复版。

    重要：
    不要随便改这些 phase 切换阈值。
    phase-aware BC 对 phase timing 很敏感。
    """

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

    action = action.cpu().numpy().astype(np.float32)
    action = np.clip(action, -1.0, 1.0)

    return action


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


def main():
    model_path = choose_model_path()

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
    ) = load_policy(model_path, device)

    env = make_env(env_config)

    success_count = 0
    episode_returns = []
    final_phases = []

    print("")
    print("=" * 60)
    print("Phase-aware BC v4 恢复版测试")
    print("=" * 60)
    print("模型:", model_path)
    print("NUM_EPISODES:", NUM_EPISODES)
    print("MAX_STEPS:", MAX_STEPS)
    print("CUBE_X_RANGE:", CUBE_X_RANGE)
    print("CUBE_Y_RANGE:", CUBE_Y_RANGE)
    print("RENDER:", RENDER)
    print("=" * 60)

    for ep in range(1, NUM_EPISODES + 1):
        obs, info = env.reset(seed=10000 + ep)

        controller = PhaseController()
        controller.reset(info)

        total_reward = 0.0
        success = False
        success_ever = False
        final_info = info

        print("")
        print("=" * 60)
        print(f"Episode {ep}/{NUM_EPISODES}")
        print("初始 cube_pos:", np.array2string(info["cube_pos"], precision=4))
        print("初始 frame_pos:", np.array2string(info["frame_pos"], precision=4))
        print("初始 phase:", controller.phase)

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

            success_ever = success_ever or bool(info["is_success"])

            if step % 20 == 0:
                print(
                    f"step={step:03d}, "
                    f"phase={phase:14s}, "
                    f"cube={np.array2string(info['cube_pos'], precision=3)}, "
                    f"tcp={np.array2string(info['tcp_pos'], precision=3)}, "
                    f"finger={float(obs[22]):+.3f}, "
                    f"lifter={float(obs[23]):+.3f}, "
                    f"lifted={info['cube_lifted']}, "
                    f"success={info['is_success']}"
                )

            # 不因为 env terminated 直接 break。
            # 等完整 release / done。
            if controller.phase == "done" and controller.phase_steps >= 5:
                success = bool(success_ever or info["is_success"])
                print(
                    f"Episode {ep} 完整结束，"
                    f"step={step}, phase={phase}, success={success}"
                )
                break

            if truncated:
                success = bool(success_ever or info["is_success"])
                print(
                    f"Episode {ep} truncated，"
                    f"step={step}, phase={phase}, success={success}"
                )
                break

        if success:
            success_count += 1

        episode_returns.append(total_reward)
        final_phases.append(controller.phase)

        final_cube = final_info["cube_pos"]
        final_frame = final_info["frame_pos"]

        xy_dist = float(np.linalg.norm(final_cube[:2] - final_frame[:2]))
        z_margin = float(final_cube[2] - final_frame[2])

        print("最终 phase:", controller.phase)
        print("最终 cube_pos:", np.array2string(final_cube, precision=4))
        print("最终 frame_pos:", np.array2string(final_frame, precision=4))
        print("final xy_dist:", round(xy_dist, 4))
        print("final z_margin:", round(z_margin, 4))
        print("success_ever:", success_ever)
        print("episode return:", total_reward)
        print("success:", success)

    env.close()

    print("")
    print("=" * 60)
    print("Phase-aware BC v4 恢复版测试总结")
    print("=" * 60)
    print(f"成功次数：{success_count}/{NUM_EPISODES}")
    print(f"成功率：{success_count / NUM_EPISODES * 100:.1f}%")
    print(f"平均 return：{np.mean(episode_returns):.3f}")

    print("")
    print("final phase 统计：")
    for phase in sorted(set(final_phases)):
        print(f"{phase}: {final_phases.count(phase)}")


if __name__ == "__main__":
    main()
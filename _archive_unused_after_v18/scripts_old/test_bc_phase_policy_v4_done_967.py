from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from openarm_pickplace_env import OpenArmPickPlaceEnv
import right_pick_place as pp


# ============================================================
# 路径和测试参数
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "bc_policy_phase_v4.pt"

NUM_EPISODES = 30
MAX_STEPS = 260

RENDER = False


# ============================================================
# 网络结构，必须和 train_bc_phase.py 一致
# ============================================================

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


def load_policy(path, device):
    checkpoint = safe_torch_load(path, device)

    input_dim = checkpoint["input_dim"]
    act_dim = checkpoint["act_dim"]
    hidden_dim = checkpoint["hidden_dim"]

    model = PhaseBCPolicy(
        input_dim=input_dim,
        act_dim=act_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    obs_aug_mean = checkpoint["obs_aug_mean"].to(device)
    obs_aug_std = checkpoint["obs_aug_std"].to(device)
    action_mean = checkpoint["action_mean"].to(device)
    action_std = checkpoint["action_std"].to(device)

    phase_names = list(checkpoint["phase_names"])
    env_config = checkpoint.get("env_config", {})

    print("已加载模型：", path)
    print("checkpoint epoch:", checkpoint.get("epoch"))
    print("best_val_loss:", checkpoint.get("best_val_loss"))
    print("phase_names:", phase_names)
    print("env_config:", env_config)

    return model, obs_aug_mean, obs_aug_std, action_mean, action_std, phase_names, env_config


def phase_onehot(phase_name, phase_names):
    onehot = np.zeros((len(phase_names),), dtype=np.float32)

    if phase_name not in phase_names:
        raise ValueError(f"未知 phase: {phase_name}")

    onehot[phase_names.index(phase_name)] = 1.0
    return onehot


def policy_action(
    model,
    obs,
    phase_name,
    phase_names,
    obs_aug_mean,
    obs_aug_std,
    action_mean,
    action_std,
    device,
):
    p = phase_onehot(phase_name, phase_names)
    obs_aug = np.concatenate([obs.astype(np.float32), p], axis=0)

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
        cube_x_range=(0.49, 0.55),
        cube_y_range=(-0.04, 0.04),
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

    print("\n测试 env 参数：")
    print("frame_skip =", env.frame_skip)
    print("max_tcp_delta =", env.max_tcp_delta)
    print("max_finger_delta =", env.max_finger_delta)
    print("max_lifter_delta =", env.max_lifter_delta)

    return env


# ============================================================
# Phase controller：只负责高层阶段切换
# 神经网络负责每个阶段里的连续动作
# ============================================================

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


# ============================================================
# 主程序
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用 device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    (
        model,
        obs_aug_mean,
        obs_aug_std,
        action_mean,
        action_std,
        phase_names,
        env_config,
    ) = load_policy(MODEL_PATH, device)

    env = make_env(env_config)

    controller = PhaseController()

    success_count = 0
    episode_returns = []
    final_phases = []

    for ep in range(1, NUM_EPISODES + 1):
        obs, info = env.reset(seed=9000 + ep)
        controller.reset(info)

        total_reward = 0.0
        success = False

        print("\n==================================================")
        print(f"Phase-BC 测试 Episode {ep}/{NUM_EPISODES}")
        print("初始 cube_pos:", info["cube_pos"])
        print("初始 tcp_pos:", info["tcp_pos"])

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
            total_reward += reward

            controller.update(obs, info)

            if step % 20 == 0:
                print(
                    f"step={step:03d}, "
                    f"phase={phase:14s}, "
                    f"reward={reward:.3f}, "
                    f"is_success={info['is_success']}, "
                    f"cube_lifted={info['cube_lifted']}, "
                    f"ik_error={info['ik_error']:.4f}, "
                    f"action={np.array2string(action, precision=2, suppress_small=True)}"
                )

            if terminated:
                success = True
                pass
            
            if controller.phase == "done":
                success = bool(info["is_success"])
                print(f"Episode {ep} 完整结束，step={step}, phase={phase}, success={success}")
                break

            if truncated:
                print(f"Episode {ep} 截断，step={step}, phase={phase}")
                break

        if success:
            success_count += 1

        episode_returns.append(total_reward)
        final_phases.append(controller.phase)

        print("最终 phase:", controller.phase)
        print("最终 cube_pos:", info["cube_pos"])
        print("最终 frame_pos:", info["frame_pos"])
        print("episode return:", total_reward)
        print("success:", success)

    env.close()

    print("\n==================================================")
    print("Phase-aware BC 测试总结")
    print("==================================================")
    print(f"成功次数：{success_count}/{NUM_EPISODES}")
    print(f"成功率：{success_count / NUM_EPISODES * 100:.1f}%")
    print(f"平均 return：{np.mean(episode_returns):.3f}")

    print("\nfinal phase 统计：")
    for phase in sorted(set(final_phases)):
        print(f"{phase}: {final_phases.count(phase)}")


if __name__ == "__main__":
    main()
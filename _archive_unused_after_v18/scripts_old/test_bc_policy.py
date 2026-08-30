from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from openarm_pickplace_env import OpenArmPickPlaceEnv


# ============================================================
# 路径和测试参数
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "bc_policy_v3.pt"

NUM_EPISODES = 20
MAX_STEPS = 220

RENDER = True


# ============================================================
# 网络结构，必须和 train_bc.py 一致
# ============================================================

class BCPolicy(nn.Module):
    def __init__(self, obs_dim=24, act_dim=5, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, obs):
        return self.net(obs)


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_policy(path, device):
    checkpoint = safe_torch_load(path, device)

    obs_dim = checkpoint["obs_dim"]
    act_dim = checkpoint["act_dim"]
    hidden_dim = checkpoint["hidden_dim"]

    model = BCPolicy(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    obs_mean = checkpoint["obs_mean"].to(device)
    obs_std = checkpoint["obs_std"].to(device)
    action_mean = checkpoint["action_mean"].to(device)
    action_std = checkpoint["action_std"].to(device)

    env_config = checkpoint.get("env_config", {})

    print("已加载模型：", path)
    print("checkpoint epoch:", checkpoint.get("epoch"))
    print("best_val_loss:", checkpoint.get("best_val_loss"))
    print("env_config:", env_config)

    return model, obs_mean, obs_std, action_mean, action_std, env_config


def policy_action(model, obs, obs_mean, obs_std, action_mean, action_std, device):
    obs_tensor = torch.from_numpy(obs).float().to(device)

    obs_norm = (obs_tensor - obs_mean) / obs_std

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

    # 必须和采集数据时的 action scale 一致
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用 device:", device)

    model, obs_mean, obs_std, action_mean, action_std, env_config = load_policy(
        MODEL_PATH,
        device,
    )

    env = make_env(env_config)

    success_count = 0
    episode_returns = []

    for ep in range(1, NUM_EPISODES + 1):
        obs, info = env.reset(seed=8000 + ep)

        total_reward = 0.0
        success = False

        print("\n==================================================")
        print(f"BC 测试 Episode {ep}/{NUM_EPISODES}")
        print("初始 cube_pos:", info["cube_pos"])
        print("初始 tcp_pos:", info["tcp_pos"])

        for step in range(MAX_STEPS):
            action = policy_action(
                model,
                obs,
                obs_mean,
                obs_std,
                action_mean,
                action_std,
                device,
            )

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if step % 20 == 0:
                print(
                    f"step={step:03d}, "
                    f"reward={reward:.3f}, "
                    f"is_success={info['is_success']}, "
                    f"cube_lifted={info['cube_lifted']}, "
                    f"ik_error={info['ik_error']:.4f}, "
                    f"action={np.array2string(action, precision=2, suppress_small=True)}"
                )

            if terminated:
                success = True
                print(f"Episode {ep} 成功，step={step}")
                break

            if truncated:
                print(f"Episode {ep} 截断，step={step}")
                break

        if success:
            success_count += 1

        episode_returns.append(total_reward)

        print("最终 cube_pos:", info["cube_pos"])
        print("最终 frame_pos:", info["frame_pos"])
        print("episode return:", total_reward)
        print("success:", success)

    env.close()

    print("\n==================================================")
    print("BC 测试总结")
    print("==================================================")
    print(f"成功次数：{success_count}/{NUM_EPISODES}")
    print(f"成功率：{success_count / NUM_EPISODES * 100:.1f}%")
    print(f"平均 return：{np.mean(episode_returns):.3f}")


if __name__ == "__main__":
    main()
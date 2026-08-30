from pathlib import Path

import numpy as np
import torch

from stable_baselines3 import PPO

from openarm_phase_rl_env import OpenArmPhaseResidualEnv


ROOT = Path(__file__).resolve().parents[1]

BC_MODEL_PATH = ROOT / "bc_policy_phase_v4.pt"
PPO_MODEL_PATH = ROOT / "ppo_phase_models" / "ppo_phase_residual_v2_final.zip"

NUM_EPISODES = 30
MAX_EPISODE_STEPS = 260

RENDER = False

RESIDUAL_SCALE = 0.05

CUBE_X_RANGE = (0.48, 0.57)
CUBE_Y_RANGE = (-0.06, 0.06)


def make_env():
    env = OpenArmPhaseResidualEnv(
        bc_model_path=BC_MODEL_PATH,
        residual_scale=RESIDUAL_SCALE,
        render_mode="human" if RENDER else None,
        randomize_cube=True,
        cube_x_range=CUBE_X_RANGE,
        cube_y_range=CUBE_Y_RANGE,
        max_episode_steps=MAX_EPISODE_STEPS,
    )
    return env


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("使用 device:", device)

    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    if not PPO_MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到 PPO 模型：{PPO_MODEL_PATH}")

    env = make_env()

    model = PPO.load(
        str(PPO_MODEL_PATH),
        env=env,
        device=device,
    )

    success_count = 0
    episode_returns = []
    final_phases = []

    for ep in range(1, NUM_EPISODES + 1):
        obs, info = env.reset(seed=11000 + ep)

        total_reward = 0.0
        success = False

        print("")
        print("==================================================")
        print(f"PPO residual v2 测试 Episode {ep}/{NUM_EPISODES}")
        print("初始 cube_pos:", info["cube_pos"])
        print("初始 phase:", info["phase"])
        print("residual_scale:", info["residual_scale"])

        for step in range(MAX_EPISODE_STEPS):
            action, _state = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

            if step % 20 == 0:
                print(
                    f"step={step:03d}, "
                    f"phase={info['phase']:14s}, "
                    f"reward={reward:.3f}, "
                    f"is_success={info['is_success']}, "
                    f"cube_lifted={info['cube_lifted']}, "
                    f"residual={np.array2string(info['residual_action'], precision=2, suppress_small=True)}, "
                    f"actual={np.array2string(info['actual_action'], precision=2, suppress_small=True)}"
                )

            if terminated:
                success = bool(info["is_success"])
                print(f"Episode {ep} terminated, step={step}, success={success}")
                break

            if truncated:
                success = bool(info["is_success"])
                print(f"Episode {ep} truncated, step={step}, success={success}")
                break

        if success:
            success_count += 1

        episode_returns.append(total_reward)
        final_phases.append(info["phase"])

        print("最终 phase:", info["phase"])
        print("最终 cube_pos:", info["cube_pos"])
        print("最终 frame_pos:", info["frame_pos"])
        print("episode return:", total_reward)
        print("success:", success)

    env.close()

    print("")
    print("==================================================")
    print("PPO residual v2 测试总结")
    print("==================================================")
    print(f"成功次数：{success_count}/{NUM_EPISODES}")
    print(f"成功率：{success_count / NUM_EPISODES * 100:.1f}%")
    print(f"平均 return：{np.mean(episode_returns):.3f}")

    print("")
    print("final phase 统计：")
    for phase in sorted(set(final_phases)):
        print(f"{phase}: {final_phases.count(phase)}")


if __name__ == "__main__":
    main()
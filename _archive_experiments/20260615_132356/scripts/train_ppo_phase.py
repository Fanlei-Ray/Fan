from pathlib import Path

import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from openarm_phase_rl_env import OpenArmPhaseResidualEnv


ROOT = Path(__file__).resolve().parents[1]

BC_MODEL_PATH = ROOT / "bc_policy_phase_v4.pt"

MODEL_DIR = ROOT / "ppo_phase_models"
LOG_DIR = ROOT / "ppo_phase_logs"

FINAL_MODEL_PATH = MODEL_DIR / "ppo_phase_residual_v2_final"

TOTAL_TIMESTEPS = 30_000

RESIDUAL_SCALE = 0.05

CUBE_X_RANGE = (0.49, 0.55)
CUBE_Y_RANGE = (-0.04, 0.04)

MAX_EPISODE_STEPS = 260


def make_env():
    env = OpenArmPhaseResidualEnv(
        bc_model_path=BC_MODEL_PATH,
        residual_scale=RESIDUAL_SCALE,
        render_mode=None,
        randomize_cube=True,
        cube_x_range=CUBE_X_RANGE,
        cube_y_range=CUBE_Y_RANGE,
        max_episode_steps=MAX_EPISODE_STEPS,
    )

    env = Monitor(env)
    return env


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("使用 device:", device)

    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    env = make_env()

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=str(MODEL_DIR),
        name_prefix="ppo_phase_residual_v2",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],
            vf=[256, 256],
        )
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=8,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        tensorboard_log=str(LOG_DIR),
    )

    print("")
    print("开始 PPO residual v2 微调")
    print(f"TOTAL_TIMESTEPS = {TOTAL_TIMESTEPS}")
    print(f"RESIDUAL_SCALE = {RESIDUAL_SCALE}")
    print(f"CUBE_X_RANGE = {CUBE_X_RANGE}")
    print(f"CUBE_Y_RANGE = {CUBE_Y_RANGE}")
    print(f"MAX_EPISODE_STEPS = {MAX_EPISODE_STEPS}")
    print("")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        progress_bar=False,
    )

    model.save(str(FINAL_MODEL_PATH))

    env.close()

    print("")
    print("PPO v2 训练完成")
    print("模型已保存：", FINAL_MODEL_PATH.with_suffix(".zip"))


if __name__ == "__main__":
    main()
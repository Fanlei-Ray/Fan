from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from openarm_right_pick_place_env_v16 import OpenArmRightPickPlaceEnv, RightPickPlaceEnvConfig


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "outputs" / "rl_right_pick_place_v16_reward_fix" / "ppo_right_pick_place_v16_final.zip"
DEFAULT_VEC = ROOT / "outputs" / "rl_right_pick_place_v16_reward_fix" / "vecnormalize_v16_final.pkl"


def parse_range(text: str):
    a, b = text.split(",")
    return float(a), float(b)


def parse_args():
    parser = argparse.ArgumentParser(description="Compact test for V16 reward-fixed OpenArm right-arm PPO.")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--vecnormalize", type=str, default=str(DEFAULT_VEC))
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--fixed-cube", action="store_true")
    parser.add_argument("--cube-x-range", type=str, default="0.49,0.55")
    parser.add_argument("--cube-y-range", type=str, default="0.02,0.08")
    parser.add_argument("--max-steps", type=int, default=450)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def make_env(args):
    x_range = parse_range(args.cube_x_range)
    y_range = parse_range(args.cube_y_range)
    cfg = RightPickPlaceEnvConfig(
        randomize_cube=not args.fixed_cube,
        max_steps=args.max_steps,
        cube_x_range=x_range,
        cube_y_range=y_range,
        timeout_penalty=12.0,
        success_bonus=120.0,
    )
    return OpenArmRightPickPlaceEnv(cfg)


def main():
    args = parse_args()
    model_path = Path(args.model)
    vec_path = Path(args.vecnormalize)
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    raw_env = make_env(args)
    vec_env = DummyVecEnv([lambda: raw_env])
    if vec_path.exists():
        print("[TEST V16] load VecNormalize:", vec_path)
        vec_env = VecNormalize.load(str(vec_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        print("[TEST V16] WARNING: vecnormalize not found, using raw obs:", vec_path)

    model = PPO.load(str(model_path), env=vec_env, device="cpu")

    successes = 0
    lifted = 0
    rewards = []
    steps = []
    dangerous = 0
    final_tcp_cube = []
    final_cube_frame = []
    timeouts = 0

    for ep in range(args.episodes):
        obs = vec_env.reset()
        done = False
        total_reward = 0.0
        last_info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = vec_env.step(action)
            done = bool(dones[0])
            total_reward += float(reward[0])
            last_info = infos[0]

        success = bool(last_info.get("place_success", False))
        ever_lifted = bool(last_info.get("ever_lifted", False))
        timeout = int(last_info.get("step_count", args.max_steps)) >= args.max_steps and not success
        successes += int(success)
        lifted += int(ever_lifted)
        timeouts += int(timeout)
        rewards.append(total_reward)
        steps.append(int(last_info.get("step_count", args.max_steps)))
        dangerous += int(last_info.get("dangerous_contacts", 0))
        final_tcp_cube.append(float(last_info.get("dist_tcp_cube", np.nan)))
        final_cube_frame.append(float(last_info.get("dist_cube_frame", np.nan)))

        print(
            f"episode={ep+1:03d} "
            f"success={success} lifted={ever_lifted} timeout={timeout} "
            f"reward={total_reward:.2f} steps={steps[-1]} "
            f"tcp_cube={final_tcp_cube[-1]:.4f} cube_frame={final_cube_frame[-1]:.4f} "
            f"lift_delta={float(last_info.get('lift_delta', 0.0)):.4f} "
            f"dangerous={int(last_info.get('dangerous_contacts', 0))}"
        )

    print("\n" + "=" * 80)
    print("[TEST V16] SUMMARY")
    print("model:", model_path)
    print("vecnormalize:", vec_path)
    print("fixed_cube:", args.fixed_cube)
    print("cube_x_range:", args.cube_x_range)
    print("cube_y_range:", args.cube_y_range)
    print(f"success_rate={successes}/{args.episodes} = {successes / max(args.episodes, 1):.3f}")
    print(f"lift_rate={lifted}/{args.episodes} = {lifted / max(args.episodes, 1):.3f}")
    print(f"timeout_rate={timeouts}/{args.episodes} = {timeouts / max(args.episodes, 1):.3f}")
    print(f"avg_reward={float(np.mean(rewards)):.3f}")
    print(f"avg_steps={float(np.mean(steps)):.1f}")
    print(f"total_dangerous_contacts={dangerous}")
    print("=" * 80)


if __name__ == "__main__":
    main()

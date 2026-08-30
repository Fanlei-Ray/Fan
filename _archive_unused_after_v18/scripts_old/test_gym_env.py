import time
import numpy as np

from openarm_pickplace_env import OpenArmPickPlaceEnv


def main():
    env = OpenArmPickPlaceEnv(
        render_mode=None,
        randomize_cube=True,
        cube_x_range=(0.49, 0.55),
        cube_y_range=(-0.04, 0.04),
        max_episode_steps=50,
    )

    obs, info = env.reset(seed=42)

    print("环境 reset 成功")
    print("obs shape:", obs.shape)
    print("初始 info:")
    for k, v in info.items():
        print(k, "=", v)

    total_reward = 0.0

    print("\n开始随机 action 测试。注意：随机 action 不会稳定完成任务，只是测试 Gym 环境接口是否正常。")

    for step in range(20):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(
            f"step={step:02d}, "
            f"reward={reward:.4f}, "
            f"total_reward={total_reward:.4f}, "
            f"terminated={terminated}, "
            f"truncated={truncated}, "
            f"cube_lifted={info['cube_lifted']}, "
            f"is_success={info['is_success']}, "
            f"ik_error={info['ik_error']:.4f}"
        )

        if terminated or truncated:
            break

    env.close()

    print("\nGymnasium 环境测试结束。")


if __name__ == "__main__":
    main()
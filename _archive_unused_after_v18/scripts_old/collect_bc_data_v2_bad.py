from pathlib import Path

import numpy as np

from openarm_pickplace_env import OpenArmPickPlaceEnv
import right_pick_place as pp


# ============================================================
# 数据采集参数
# ============================================================

NUM_SUCCESS_DEMOS = 100
MAX_ATTEMPTS = 300
MAX_STEPS_PER_EPISODE = 240

SAVE_PATH = Path(__file__).resolve().parents[1] / "bc_dataset_v2.npz"

CUBE_X_RANGE = (0.49, 0.55)
CUBE_Y_RANGE = (-0.04, 0.04)


# ============================================================
# Expert policy：手写规则策略，输出 Gym action
# action = [dx, dy, dz, gripper, lifter]
# ============================================================

class ExpertPickPlacePolicy:
    def __init__(self, env):
        self.env = env
        self.phase = "open_gripper"
        self.phase_steps = 0
        self.pick_cube_pos = None

    def reset(self):
        self.phase = "open_gripper"
        self.phase_steps = 0
        self.pick_cube_pos = None

    def _move_tcp_action(self, tcp_pos, target_pos):
        """
        把 TCP 当前位置到目标点的差值，转换成 Gym action 的 dx/dy/dz。
        env.step 里会再乘 env.max_tcp_delta。
        """
        delta = target_pos - tcp_pos
        action_xyz = delta / self.env.max_tcp_delta
        return np.clip(action_xyz, -1.0, 1.0)

    def _phase_done(self, tcp_pos, target_pos, threshold):
        return np.linalg.norm(tcp_pos - target_pos) < threshold

    def act(self, obs, info):
        tcp_pos = info["tcp_pos"]
        cube_pos = info["cube_pos"]
        frame_pos = info["frame_pos"]

        finger_ctrl = obs[22]
        lifter_ctrl = obs[23]

        if self.pick_cube_pos is None:
            # 关键修正：
            # 抓取阶段固定最开始看到的方块位置。
            # 不要在方块被碰歪后继续追着它跑，否则会越推越远。
            self.pick_cube_pos = cube_pos.copy()

        action = np.zeros(5, dtype=np.float32)
        self.phase_steps += 1

        # --------------------------------------------------------
        # 0. 张开夹爪
        # --------------------------------------------------------
        if self.phase == "open_gripper":
            action[3] = +1.0

            if finger_ctrl >= pp.LEFT_FINGER_PRE_OPEN - 0.015 or self.phase_steps > 25:
                self.phase = "move_pregrasp"
                self.phase_steps = 0

            return action

        # --------------------------------------------------------
        # 1. 移到方块上方
        # --------------------------------------------------------
        if self.phase == "move_pregrasp":
            target = self.pick_cube_pos + pp.PICK_PREGRASP_OFFSET
            action[:3] = self._move_tcp_action(tcp_pos, target)

            # 保持夹爪微开
            action[3] = +0.15
            action[4] = 0.0

            if self._phase_done(tcp_pos, target, threshold=0.015) or self.phase_steps > 80:
                self.phase = "move_grasp"
                self.phase_steps = 0

            return action

        # --------------------------------------------------------
        # 2. 慢慢下探到抓取位置
        # --------------------------------------------------------
        if self.phase == "move_grasp":
            target = self.pick_cube_pos + pp.PICK_GRASP_OFFSET
            action[:3] = self._move_tcp_action(tcp_pos, target)

            # 下探时仍保持夹爪微开，不提前夹
            action[3] = +0.05
            action[4] = 0.0

            if self._phase_done(tcp_pos, target, threshold=0.012) or self.phase_steps > 90:
                self.phase = "close_gripper"
                self.phase_steps = 0

            return action

        # --------------------------------------------------------
        # 3. 慢慢闭合夹爪
        # --------------------------------------------------------
        if self.phase == "close_gripper":
            action[:3] = 0.0
            action[3] = -0.45
            action[4] = 0.0

            if finger_ctrl <= pp.LEFT_FINGER_CLOSE + 0.025 or self.phase_steps > 55:
                self.phase = "hold_grasp"
                self.phase_steps = 0

            return action

        # --------------------------------------------------------
        # 4. 夹住后原地保持，让接触稳定
        # --------------------------------------------------------
        if self.phase == "hold_grasp":
            action[:3] = 0.0
            action[3] = -0.25
            action[4] = 0.0

            if self.phase_steps > 12:
                self.phase = "lift_object"
                self.phase_steps = 0

            return action

        # --------------------------------------------------------
        # 5. 抬升 lifter
        # --------------------------------------------------------
        if self.phase == "lift_object":
            action[:3] = 0.0
            action[3] = -0.25
            action[4] = +1.0

            if info["cube_lifted"] or lifter_ctrl >= pp.LIFTER_UP - 0.015 or self.phase_steps > 45:
                self.phase = "move_preplace"
                self.phase_steps = 0

            return action

        # --------------------------------------------------------
        # 6. 移到黑框上方
        # --------------------------------------------------------
        if self.phase == "move_preplace":
            target = frame_pos + pp.PLACE_PREPLACE_OFFSET
            action[:3] = self._move_tcp_action(tcp_pos, target)

            # 移动过程中保持夹紧
            action[3] = -0.25
            action[4] = 0.0

            if self._phase_done(tcp_pos, target, threshold=0.035) or self.phase_steps > 100:
                self.phase = "move_release"
                self.phase_steps = 0

            return action

        # --------------------------------------------------------
        # 7. 下降到释放位置
        # --------------------------------------------------------
        if self.phase == "move_release":
            target = frame_pos + pp.PLACE_RELEASE_OFFSET
            action[:3] = self._move_tcp_action(tcp_pos, target)

            action[3] = -0.20
            action[4] = 0.0

            if self._phase_done(tcp_pos, target, threshold=0.030) or self.phase_steps > 80:
                self.phase = "open_release"
                self.phase_steps = 0

            return action

        # --------------------------------------------------------
        # 8. 张开释放
        # --------------------------------------------------------
        if self.phase == "open_release":
            action[:3] = 0.0
            action[3] = +1.0
            action[4] = 0.0

            return action

        return action


# ============================================================
# 单条 episode 采集
# ============================================================

def collect_one_episode(env, seed=None):
    obs, info = env.reset(seed=seed)

    expert = ExpertPickPlacePolicy(env)
    expert.reset()

    obs_list = []
    action_list = []
    reward_list = []

    success = False

    for step in range(MAX_STEPS_PER_EPISODE):
        action = expert.act(obs, info)

        obs_list.append(obs.copy())
        action_list.append(action.copy())

        obs, reward, terminated, truncated, info = env.step(action)
        reward_list.append(reward)

        if terminated:
            success = True
            break

        if truncated:
            break

    return {
        "success": success,
        "obs": np.asarray(obs_list, dtype=np.float32),
        "actions": np.asarray(action_list, dtype=np.float32),
        "rewards": np.asarray(reward_list, dtype=np.float32),
        "final_info": info,
        "final_phase": expert.phase,
    }


# ============================================================
# 主程序
# ============================================================

def main():
    env = OpenArmPickPlaceEnv(
        render_mode=None,
        randomize_cube=True,
        cube_x_range=CUBE_X_RANGE,
        cube_y_range=CUBE_Y_RANGE,
        max_episode_steps=MAX_STEPS_PER_EPISODE,
    )

    # 采集示教时动作放慢，减少把方块推飞
    env.max_tcp_delta = 0.018
    env.max_finger_delta = 0.015
    env.max_lifter_delta = 0.012

    all_obs = []
    all_actions = []
    all_rewards = []
    episode_lengths = []

    success_count = 0
    attempt_count = 0

    print("开始采集 BC 示教数据 v2")
    print(f"目标成功 demo 数：{NUM_SUCCESS_DEMOS}")
    print(f"最大尝试次数：{MAX_ATTEMPTS}")
    print(f"MAX_STEPS_PER_EPISODE：{MAX_STEPS_PER_EPISODE}")
    print(f"CUBE_X_RANGE：{CUBE_X_RANGE}")
    print(f"CUBE_Y_RANGE：{CUBE_Y_RANGE}")
    print(f"保存路径：{SAVE_PATH}")
    print("")
    print("env action scale:")
    print(f"env.max_tcp_delta = {env.max_tcp_delta}")
    print(f"env.max_finger_delta = {env.max_finger_delta}")
    print(f"env.max_lifter_delta = {env.max_lifter_delta}")

    while success_count < NUM_SUCCESS_DEMOS and attempt_count < MAX_ATTEMPTS:
        attempt_count += 1

        result = collect_one_episode(env, seed=2000 + attempt_count)

        if result["success"]:
            success_count += 1

            all_obs.append(result["obs"])
            all_actions.append(result["actions"])
            all_rewards.append(result["rewards"])
            episode_lengths.append(len(result["obs"]))

            print(
                f"[成功] demo {success_count:03d}/{NUM_SUCCESS_DEMOS}, "
                f"attempt={attempt_count}, "
                f"steps={len(result['obs'])}, "
                f"phase={result['final_phase']}, "
                f"cube_pos={result['final_info']['cube_pos']}, "
                f"cube_lifted={result['final_info']['cube_lifted']}, "
                f"is_success={result['final_info']['is_success']}"
            )
        else:
            print(
                f"[失败] attempt={attempt_count}, "
                f"steps={len(result['obs'])}, "
                f"phase={result['final_phase']}, "
                f"cube_pos={result['final_info']['cube_pos']}, "
                f"cube_lifted={result['final_info']['cube_lifted']}, "
                f"is_success={result['final_info']['is_success']}"
            )

    env.close()

    if success_count == 0:
        print("")
        print("没有采集到成功 demo，先不要训练 BC。")
        return

    obs_array = np.concatenate(all_obs, axis=0)
    action_array = np.concatenate(all_actions, axis=0)
    reward_array = np.concatenate(all_rewards, axis=0)
    episode_lengths = np.asarray(episode_lengths, dtype=np.int32)

    np.savez(
        SAVE_PATH,
        obs=obs_array,
        actions=action_array,
        rewards=reward_array,
        episode_lengths=episode_lengths,
        cube_x_range=np.asarray(CUBE_X_RANGE, dtype=np.float32),
        cube_y_range=np.asarray(CUBE_Y_RANGE, dtype=np.float32),
        max_tcp_delta=np.asarray([env.max_tcp_delta], dtype=np.float32),
        max_finger_delta=np.asarray([env.max_finger_delta], dtype=np.float32),
        max_lifter_delta=np.asarray([env.max_lifter_delta], dtype=np.float32),
    )

    print("")
    print("==================================================")
    print("采集完成")
    print("==================================================")
    print(f"成功 demo 数：{success_count}")
    print(f"尝试次数：{attempt_count}")
    print(f"总样本数：{len(obs_array)}")
    print(f"obs shape：{obs_array.shape}")
    print(f"actions shape：{action_array.shape}")
    print(f"rewards shape：{reward_array.shape}")
    print(f"episode_lengths shape：{episode_lengths.shape}")
    print(f"平均 episode 长度：{episode_lengths.mean():.1f}")
    print(f"最短 episode 长度：{episode_lengths.min()}")
    print(f"最长 episode 长度：{episode_lengths.max()}")
    print(f"已保存：{SAVE_PATH}")


if __name__ == "__main__":
    main()
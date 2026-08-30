from pathlib import Path

import numpy as np

from openarm_pickplace_env import OpenArmPickPlaceEnv
import right_pick_place as pp


# ============================================================
# 数据采集参数 v3
# ============================================================

NUM_SUCCESS_DEMOS = 100
MAX_ATTEMPTS = 300
MAX_STEPS_PER_EPISODE = 420

SAVE_PATH = Path(__file__).resolve().parents[1] / "bc_dataset_v3.npz"

CUBE_X_RANGE = (0.49, 0.55)
CUBE_Y_RANGE = (-0.04, 0.04)


# ============================================================
# Expert policy v3
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
        delta = target_pos - tcp_pos
        action_xyz = delta / self.env.max_tcp_delta
        return np.clip(action_xyz, -1.0, 1.0)

    def _phase_done(self, tcp_pos, target_pos, threshold):
        return np.linalg.norm(tcp_pos - target_pos) < threshold

    def act(self, obs, info):
        tcp_pos = info["tcp_pos"]
        cube_pos = info["cube_pos"]
        frame_pos = info["frame_pos"]

        finger_ctrl = float(obs[22])
        lifter_ctrl = float(obs[23])

        if self.pick_cube_pos is None:
            # 抓取阶段固定最初方块位置，避免方块被轻微碰动后追着它跑
            self.pick_cube_pos = cube_pos.copy()

        action = np.zeros(5, dtype=np.float32)
        self.phase_steps += 1

        # --------------------------------------------------------
        # 0. 张开夹爪
        # --------------------------------------------------------
        if self.phase == "open_gripper":
            action[3] = +1.0

            if finger_ctrl >= pp.LEFT_FINGER_PRE_OPEN - 0.02 or self.phase_steps > 18:
                self.phase = "move_pregrasp"
                self.phase_steps = 0

            return action

        # --------------------------------------------------------
        # 1. 移动到方块上方
        # --------------------------------------------------------
        if self.phase == "move_pregrasp":
            target = self.pick_cube_pos + pp.PICK_PREGRASP_OFFSET

            action[:3] = self._move_tcp_action(tcp_pos, target)
            action[3] = +0.2
            action[4] = 0.0

            if self._phase_done(tcp_pos, target, threshold=0.025) or self.phase_steps > 80:
                self.phase = "move_grasp"
                self.phase_steps = 0

            return action

        # --------------------------------------------------------
        # 2. 下探到抓取位置
        # --------------------------------------------------------
        if self.phase == "move_grasp":
            target = self.pick_cube_pos + pp.PICK_GRASP_OFFSET

            action[:3] = self._move_tcp_action(tcp_pos, target)
            action[3] = +0.1
            action[4] = 0.0

            if self._phase_done(tcp_pos, target, threshold=0.020) or self.phase_steps > 90:
                self.phase = "close_gripper"
                self.phase_steps = 0

            return action

        # --------------------------------------------------------
        # 3. 闭合夹爪
        # --------------------------------------------------------
        if self.phase == "close_gripper":
            action[:3] = 0.0
            action[3] = -1.0
            action[4] = 0.0

            if finger_ctrl <= pp.LEFT_FINGER_CLOSE + 0.035 or self.phase_steps > 35:
                self.phase = "hold_grasp"
                self.phase_steps = 0

            return action

        # --------------------------------------------------------
        # 4. 闭合后保持，让接触稳定
        # --------------------------------------------------------
        if self.phase == "hold_grasp":
            action[:3] = 0.0
            action[3] = -0.6
            action[4] = 0.0

            if self.phase_steps > 10:
                self.phase = "lift_object"
                self.phase_steps = 0

            return action

        # --------------------------------------------------------
        # 5. 抬升
        # --------------------------------------------------------
        if self.phase == "lift_object":
            action[:3] = 0.0
            action[3] = -0.6
            action[4] = +1.0

            if info["cube_lifted"] or lifter_ctrl >= pp.LIFTER_UP - 0.015 or self.phase_steps > 45:
                self.phase = "move_preplace"
                self.phase_steps = 0

            return action

        # --------------------------------------------------------
        # 6. 移动到黑框上方
        # --------------------------------------------------------
        if self.phase == "move_preplace":
            target = frame_pos + pp.PLACE_PREPLACE_OFFSET

            action[:3] = self._move_tcp_action(tcp_pos, target)
            action[3] = -0.5
            action[4] = 0.0

            if self._phase_done(tcp_pos, target, threshold=0.040) or self.phase_steps > 110:
                self.phase = "move_release"
                self.phase_steps = 0

            return action

        # --------------------------------------------------------
        # 7. 下降到释放位置
        # --------------------------------------------------------
        if self.phase == "move_release":
            target = frame_pos + pp.PLACE_RELEASE_OFFSET

            action[:3] = self._move_tcp_action(tcp_pos, target)
            action[3] = -0.4
            action[4] = 0.0

            if self._phase_done(tcp_pos, target, threshold=0.035) or self.phase_steps > 90:
                self.phase = "open_release"
                self.phase_steps = 0

            return action

        # --------------------------------------------------------
        # 8. 张开夹爪释放
        # --------------------------------------------------------
        if self.phase == "open_release":
            action[:3] = 0.0
            action[3] = +1.0
            action[4] = 0.0

            if finger_ctrl >= pp.LEFT_FINGER_OPEN - 0.03 or self.phase_steps > 35:
                self.phase = "done"
                self.phase_steps = 0

            return action

        # --------------------------------------------------------
        # 9. 完成后保持
        # --------------------------------------------------------
        if self.phase == "done":
            action[:3] = 0.0
            action[3] = +0.2
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
    phase_list = []

    success = False
    final_terminated = False
    final_truncated = False

    for step in range(MAX_STEPS_PER_EPISODE):
        action = expert.act(obs, info)

        obs_list.append(obs.copy())
        action_list.append(action.copy())
        phase_list.append(expert.phase)

        obs, reward, terminated, truncated, info = env.step(action)
        reward_list.append(reward)

        final_terminated = terminated
        final_truncated = truncated

        # 重要：
        # env 的 terminated 会在“方块到黑框上方”时触发，
        # 但我们还要继续采集下降和释放动作。
        # 所以这里不因为 terminated 立即 break。
        if expert.phase == "done":
            # 再多等几步，让方块释放后稳定
            for _ in range(8):
                if len(obs_list) >= MAX_STEPS_PER_EPISODE:
                    break

                action = np.zeros(5, dtype=np.float32)
                action[3] = +0.2

                obs_list.append(obs.copy())
                action_list.append(action.copy())
                phase_list.append("done_hold")

                obs, reward, terminated, truncated, info = env.step(action)
                reward_list.append(reward)

                final_terminated = terminated
                final_truncated = truncated

            success = bool(info["is_success"])
            break

        if truncated:
            break

    return {
        "success": success,
        "obs": np.asarray(obs_list, dtype=np.float32),
        "actions": np.asarray(action_list, dtype=np.float32),
        "rewards": np.asarray(reward_list, dtype=np.float32),
        "phases": np.asarray(phase_list),
        "final_info": info,
        "final_phase": expert.phase,
        "terminated": bool(final_terminated),
        "truncated": bool(final_truncated),
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

    # v3：比 v2 快一点，避免卡在 close_gripper / 超时
    env.frame_skip = 30
    env.max_tcp_delta = 0.024
    env.max_finger_delta = 0.040
    env.max_lifter_delta = 0.025

    all_obs = []
    all_actions = []
    all_rewards = []
    all_phases = []
    episode_lengths = []

    success_count = 0
    attempt_count = 0

    print("开始采集 BC 示教数据 v3")
    print(f"目标成功 demo 数：{NUM_SUCCESS_DEMOS}")
    print(f"最大尝试次数：{MAX_ATTEMPTS}")
    print(f"MAX_STEPS_PER_EPISODE：{MAX_STEPS_PER_EPISODE}")
    print(f"CUBE_X_RANGE：{CUBE_X_RANGE}")
    print(f"CUBE_Y_RANGE：{CUBE_Y_RANGE}")
    print(f"保存路径：{SAVE_PATH}")
    print("")
    print("env action scale:")
    print(f"env.frame_skip = {env.frame_skip}")
    print(f"env.max_tcp_delta = {env.max_tcp_delta}")
    print(f"env.max_finger_delta = {env.max_finger_delta}")
    print(f"env.max_lifter_delta = {env.max_lifter_delta}")

    while success_count < NUM_SUCCESS_DEMOS and attempt_count < MAX_ATTEMPTS:
        attempt_count += 1

        result = collect_one_episode(env, seed=3000 + attempt_count)

        if result["success"]:
            success_count += 1

            all_obs.append(result["obs"])
            all_actions.append(result["actions"])
            all_rewards.append(result["rewards"])
            all_phases.append(result["phases"])
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
                f"is_success={result['final_info']['is_success']}, "
                f"terminated={result['terminated']}, "
                f"truncated={result['truncated']}"
            )

    env.close()

    if success_count == 0:
        print("")
        print("没有采集到成功 demo，先不要训练 BC。")
        return

    obs_array = np.concatenate(all_obs, axis=0)
    action_array = np.concatenate(all_actions, axis=0)
    reward_array = np.concatenate(all_rewards, axis=0)
    phase_array = np.concatenate(all_phases, axis=0)
    episode_lengths = np.asarray(episode_lengths, dtype=np.int32)

    np.savez(
        SAVE_PATH,
        obs=obs_array,
        actions=action_array,
        rewards=reward_array,
        phases=phase_array,
        episode_lengths=episode_lengths,
        cube_x_range=np.asarray(CUBE_X_RANGE, dtype=np.float32),
        cube_y_range=np.asarray(CUBE_Y_RANGE, dtype=np.float32),
        max_tcp_delta=np.asarray([env.max_tcp_delta], dtype=np.float32),
        max_finger_delta=np.asarray([env.max_finger_delta], dtype=np.float32),
        max_lifter_delta=np.asarray([env.max_lifter_delta], dtype=np.float32),
        frame_skip=np.asarray([env.frame_skip], dtype=np.int32),
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
    print(f"phases shape：{phase_array.shape}")
    print(f"episode_lengths shape：{episode_lengths.shape}")
    print(f"平均 episode 长度：{episode_lengths.mean():.1f}")
    print(f"最短 episode 长度：{episode_lengths.min()}")
    print(f"最长 episode 长度：{episode_lengths.max()}")
    print(f"已保存：{SAVE_PATH}")


if __name__ == "__main__":
    main()
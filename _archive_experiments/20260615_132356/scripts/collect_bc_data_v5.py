from pathlib import Path
import time

import numpy as np

from openarm_pickplace_env import OpenArmPickPlaceEnv
import right_pick_place as pp


ROOT = Path(__file__).resolve().parents[1]
SAVE_PATH = ROOT / "bc_dataset_v5_1.npz"

TARGET_SUCCESS_DEMOS = 300
MAX_ATTEMPTS = 600
MAX_EPISODE_STEPS = 460

CUBE_X_RANGE = (0.48, 0.57)
CUBE_Y_RANGE = (-0.06, 0.06)

FRAME_SKIP = 30
MAX_TCP_DELTA = 0.024
MAX_FINGER_DELTA = 0.040
MAX_LIFTER_DELTA = 0.025

PRINT_EVERY_ATTEMPT = True


PHASE_NAMES = [
    "open_gripper",
    "move_pregrasp",
    "move_grasp",
    "close_gripper",
    "hold_grasp",
    "lift_object",
    "move_preplace",
    "move_release",
    "open_release",
    "done",
    "done_hold",
]


class ExpertPhasePolicy:
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

    def _action_to_targets(
        self,
        obs,
        info,
        tcp_target=None,
        finger_target=None,
        lifter_target=None,
    ):
        action = np.zeros(5, dtype=np.float32)

        tcp_pos = info["tcp_pos"]
        finger_ctrl = float(obs[22])
        lifter_ctrl = float(obs[23])

        if tcp_target is not None:
            delta = np.asarray(tcp_target, dtype=np.float32) - tcp_pos
            action[:3] = np.clip(delta / MAX_TCP_DELTA, -1.0, 1.0)

        if finger_target is not None:
            action[3] = np.clip(
                (float(finger_target) - finger_ctrl) / MAX_FINGER_DELTA,
                -1.0,
                1.0,
            )

        if lifter_target is not None:
            action[4] = np.clip(
                (float(lifter_target) - lifter_ctrl) / MAX_LIFTER_DELTA,
                -1.0,
                1.0,
            )

        return action.astype(np.float32)

    def act(self, obs, info):
        cube_pos = info["cube_pos"]
        frame_pos = info["frame_pos"]
        tcp_pos = info["tcp_pos"]

        if self.pick_cube_pos is None:
            self.pick_cube_pos = cube_pos.copy()

        if self.phase == "open_gripper":
            return self._action_to_targets(
                obs,
                info,
                tcp_target=tcp_pos,
                finger_target=pp.LEFT_FINGER_PRE_OPEN,
                lifter_target=pp.LIFTER_HOME,
            )

        if self.phase == "move_pregrasp":
            target = self.pick_cube_pos + pp.PICK_PREGRASP_OFFSET
            return self._action_to_targets(
                obs,
                info,
                tcp_target=target,
                finger_target=pp.LEFT_FINGER_PRE_OPEN,
                lifter_target=pp.LIFTER_HOME,
            )

        if self.phase == "move_grasp":
            target = self.pick_cube_pos + pp.PICK_GRASP_OFFSET
            return self._action_to_targets(
                obs,
                info,
                tcp_target=target,
                finger_target=pp.LEFT_FINGER_PRE_OPEN,
                lifter_target=pp.LIFTER_HOME,
            )

        if self.phase == "close_gripper":
            target = self.pick_cube_pos + pp.PICK_GRASP_OFFSET
            return self._action_to_targets(
                obs,
                info,
                tcp_target=target,
                finger_target=pp.LEFT_FINGER_CLOSE,
                lifter_target=pp.LIFTER_HOME,
            )

        if self.phase == "hold_grasp":
            target = self.pick_cube_pos + pp.PICK_GRASP_OFFSET
            return self._action_to_targets(
                obs,
                info,
                tcp_target=target,
                finger_target=pp.LEFT_FINGER_CLOSE,
                lifter_target=pp.LIFTER_HOME,
            )

        if self.phase == "lift_object":
            return self._action_to_targets(
                obs,
                info,
                tcp_target=tcp_pos,
                finger_target=pp.LEFT_FINGER_CLOSE,
                lifter_target=pp.LIFTER_UP,
            )

        if self.phase == "move_preplace":
            target = frame_pos + pp.PLACE_PREPLACE_OFFSET
            return self._action_to_targets(
                obs,
                info,
                tcp_target=target,
                finger_target=pp.LEFT_FINGER_CLOSE,
                lifter_target=pp.LIFTER_UP,
            )

        if self.phase == "move_release":
            target = frame_pos + pp.PLACE_RELEASE_OFFSET
            return self._action_to_targets(
                obs,
                info,
                tcp_target=target,
                finger_target=pp.LEFT_FINGER_CLOSE,
                lifter_target=pp.LIFTER_UP,
            )

        if self.phase == "open_release":
            target = frame_pos + pp.PLACE_RELEASE_OFFSET
            return self._action_to_targets(
                obs,
                info,
                tcp_target=target,
                finger_target=pp.LEFT_FINGER_OPEN,
                lifter_target=pp.LIFTER_UP,
            )

        if self.phase in ["done", "done_hold"]:
            return self._action_to_targets(
                obs,
                info,
                tcp_target=tcp_pos,
                finger_target=pp.LEFT_FINGER_OPEN,
                lifter_target=pp.LIFTER_UP,
            )

        raise RuntimeError(f"未知 phase: {self.phase}")

    def update(self, obs, info):
        self.phase_steps += 1

        tcp_pos = info["tcp_pos"]
        frame_pos = info["frame_pos"]
        finger_ctrl = float(obs[22])
        lifter_ctrl = float(obs[23])

        if self.phase == "open_gripper":
            if finger_ctrl >= pp.LEFT_FINGER_PRE_OPEN - 0.02 or self.phase_steps > 22:
                # 等方块稳定后再记录抓取目标
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
            if self.phase_steps > 6:
                self._switch("done_hold")
            return

        if self.phase == "done_hold":
            return


def make_env():
    env = OpenArmPickPlaceEnv(
        render_mode=None,
        randomize_cube=True,
        cube_x_range=CUBE_X_RANGE,
        cube_y_range=CUBE_Y_RANGE,
        max_episode_steps=MAX_EPISODE_STEPS,
    )

    env.frame_skip = FRAME_SKIP
    env.max_tcp_delta = MAX_TCP_DELTA
    env.max_finger_delta = MAX_FINGER_DELTA
    env.max_lifter_delta = MAX_LIFTER_DELTA

    return env


def main():
    env = make_env()
    expert = ExpertPhasePolicy()

    all_obs = []
    all_actions = []
    all_rewards = []
    all_phases = []
    episode_lengths = []

    success_count = 0
    attempt_count = 0

    start_time = time.time()

    print("开始采集 BC v5 数据")
    print("保存路径:", SAVE_PATH)
    print("目标成功 demo:", TARGET_SUCCESS_DEMOS)
    print("最大尝试次数:", MAX_ATTEMPTS)
    print("cube_x_range:", CUBE_X_RANGE)
    print("cube_y_range:", CUBE_Y_RANGE)
    print("")

    while success_count < TARGET_SUCCESS_DEMOS and attempt_count < MAX_ATTEMPTS:
        attempt_count += 1

        seed = 50000 + attempt_count
        obs, info = env.reset(seed=seed)
        expert.reset(info)

        ep_obs = []
        ep_actions = []
        ep_rewards = []
        ep_phases = []

        final_info = info
        finished = False

        for step in range(MAX_EPISODE_STEPS):
            phase = expert.phase
            action = expert.act(obs, info)

            next_obs, reward, terminated, truncated, next_info = env.step(action)

            ep_obs.append(obs.astype(np.float32))
            ep_actions.append(action.astype(np.float32))
            ep_rewards.append(float(reward))
            ep_phases.append(phase)

            obs = next_obs
            info = next_info
            final_info = next_info

            expert.update(obs, info)

            if expert.phase == "done_hold" and expert.phase_steps > 8:
                finished = True
                break

            if truncated:
                break

        demo_success = bool(final_info["is_success"]) and finished

        if demo_success:
            success_count += 1

            all_obs.extend(ep_obs)
            all_actions.extend(ep_actions)
            all_rewards.extend(ep_rewards)
            all_phases.extend(ep_phases)
            episode_lengths.append(len(ep_obs))

            status = "成功"
        else:
            status = "失败"

        if PRINT_EVERY_ATTEMPT:
            print(
                f"attempt {attempt_count:03d} | {status} | "
                f"success {success_count:03d}/{TARGET_SUCCESS_DEMOS} | "
                f"steps {len(ep_obs):03d} | "
                f"final_phase {expert.phase:12s} | "
                f"is_success {bool(final_info['is_success'])} | "
                f"cube_pos {np.array2string(final_info['cube_pos'], precision=3)}"
            )

    env.close()

    if success_count == 0:
        raise RuntimeError("没有采集到成功 demo，先不要训练。")

    obs_arr = np.asarray(all_obs, dtype=np.float32)
    actions_arr = np.asarray(all_actions, dtype=np.float32)
    rewards_arr = np.asarray(all_rewards, dtype=np.float32)
    phases_arr = np.asarray(all_phases, dtype="<U32")
    episode_lengths_arr = np.asarray(episode_lengths, dtype=np.int32)

    np.savez(
        SAVE_PATH,
        obs=obs_arr,
        actions=actions_arr,
        rewards=rewards_arr,
        phases=phases_arr,
        episode_lengths=episode_lengths_arr,
        phase_names=np.asarray(PHASE_NAMES, dtype="<U32"),
        cube_x_range=np.asarray(CUBE_X_RANGE, dtype=np.float32),
        cube_y_range=np.asarray(CUBE_Y_RANGE, dtype=np.float32),
        env_config={
            "frame_skip": FRAME_SKIP,
            "max_tcp_delta": MAX_TCP_DELTA,
            "max_finger_delta": MAX_FINGER_DELTA,
            "max_lifter_delta": MAX_LIFTER_DELTA,
            "max_episode_steps": MAX_EPISODE_STEPS,
            "cube_x_range": CUBE_X_RANGE,
            "cube_y_range": CUBE_Y_RANGE,
        },
    )

    elapsed = time.time() - start_time

    print("")
    print("=" * 60)
    print("BC v5 数据采集完成")
    print("=" * 60)
    print("成功 demo 数:", success_count)
    print("尝试次数:", attempt_count)
    print("总样本数:", len(obs_arr))
    print("obs shape:", obs_arr.shape)
    print("actions shape:", actions_arr.shape)
    print("rewards shape:", rewards_arr.shape)
    print("phases shape:", phases_arr.shape)
    print("episode_lengths shape:", episode_lengths_arr.shape)
    print("平均 episode 长度:", float(np.mean(episode_lengths_arr)))
    print("最短 episode 长度:", int(np.min(episode_lengths_arr)))
    print("最长 episode 长度:", int(np.max(episode_lengths_arr)))
    print("耗时 秒:", round(elapsed, 1))
    print("已保存:", SAVE_PATH)


if __name__ == "__main__":
    main()
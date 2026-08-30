from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import torch
import torch.nn as nn

from openarm_pickplace_env import OpenArmPickPlaceEnv
import right_pick_place as pp


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BC_MODEL_PATH = ROOT / "bc_policy_phase_v4.pt"


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


def phase_onehot(phase_name, phase_names):
    onehot = np.zeros((len(phase_names),), dtype=np.float32)

    if phase_name not in phase_names:
        raise ValueError(f"未知 phase: {phase_name}")

    onehot[phase_names.index(phase_name)] = 1.0
    return onehot


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


class OpenArmPhaseResidualEnv(gym.Env):
    """
    PPO residual 环境 v2。

    observation:
        base_obs(24) + phase_onehot(11) = 35

    PPO action:
        residual_action, shape=(5,)

    actual action:
        actual_action = BC_action + residual_scale * residual_action

    这版重点：
        1. residual_scale 小，默认 0.05
        2. 不让 PPO 大幅破坏 BC
        3. reward 更看重完整 done + 最终 success
    """

    metadata = {
        "render_modes": ["human"],
        "render_fps": 50,
    }

    def __init__(
        self,
        bc_model_path=DEFAULT_BC_MODEL_PATH,
        residual_scale=0.05,
        render_mode=None,
        randomize_cube=True,
        cube_x_range=(0.49, 0.55),
        cube_y_range=(-0.04, 0.04),
        max_episode_steps=260,
    ):
        super().__init__()

        self.bc_model_path = Path(bc_model_path)
        self.residual_scale = float(residual_scale)
        self.render_mode = render_mode
        self.max_episode_steps = int(max_episode_steps)

        # SB3 在 GPU 上训练 PPO，但这里 BC 推理放 CPU 即可，MuJoCo 本来也在 CPU。
        # 这样更稳，避免每个 env step 都在 CPU/GPU 间频繁切换。
        self.device = torch.device("cpu")

        self._load_bc_policy()

        self.base_env = OpenArmPickPlaceEnv(
            render_mode=render_mode,
            randomize_cube=randomize_cube,
            cube_x_range=cube_x_range,
            cube_y_range=cube_y_range,
            max_episode_steps=max_episode_steps,
        )

        # 和 BC 数据采集时保持一致
        if "frame_skip" in self.env_config:
            self.base_env.frame_skip = int(self.env_config["frame_skip"])

        if "max_tcp_delta" in self.env_config:
            self.base_env.max_tcp_delta = float(self.env_config["max_tcp_delta"])

        if "max_finger_delta" in self.env_config:
            self.base_env.max_finger_delta = float(self.env_config["max_finger_delta"])

        if "max_lifter_delta" in self.env_config:
            self.base_env.max_lifter_delta = float(self.env_config["max_lifter_delta"])

        self.phase_controller = PhaseController()

        self.obs_dim = 24
        self.phase_dim = len(self.phase_names)
        self.aug_obs_dim = self.obs_dim + self.phase_dim

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.aug_obs_dim,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32,
        )

        self.step_count = 0
        self.last_obs = None
        self.last_info = None

    def _load_bc_policy(self):
        if not self.bc_model_path.exists():
            raise FileNotFoundError(f"找不到 BC 模型：{self.bc_model_path}")

        checkpoint = safe_torch_load(self.bc_model_path, self.device)

        self.phase_names = list(checkpoint["phase_names"])
        self.env_config = checkpoint.get("env_config", {})

        self.obs_aug_mean = checkpoint["obs_aug_mean"].to(self.device)
        self.obs_aug_std = checkpoint["obs_aug_std"].to(self.device)
        self.action_mean = checkpoint["action_mean"].to(self.device)
        self.action_std = checkpoint["action_std"].to(self.device)

        self.bc_model = PhaseBCPolicy(
            input_dim=int(checkpoint["input_dim"]),
            act_dim=int(checkpoint["act_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
        ).to(self.device)

        self.bc_model.load_state_dict(checkpoint["model_state_dict"])
        self.bc_model.eval()

        print("已加载 BC base policy:", self.bc_model_path)
        print("BC best_val_loss:", checkpoint.get("best_val_loss"))
        print("BC env_config:", self.env_config)

    def _get_aug_obs(self, obs):
        phase = self.phase_controller.phase
        p = phase_onehot(phase, self.phase_names)
        return np.concatenate([obs.astype(np.float32), p], axis=0).astype(np.float32)

    def _bc_action(self, obs, phase):
        p = phase_onehot(phase, self.phase_names)
        obs_aug = np.concatenate([obs.astype(np.float32), p], axis=0).astype(np.float32)

        obs_tensor = torch.from_numpy(obs_aug).float().to(self.device)
        obs_norm = (obs_tensor - self.obs_aug_mean) / self.obs_aug_std

        with torch.no_grad():
            action_norm = self.bc_model(obs_norm.unsqueeze(0)).squeeze(0)
            action = action_norm * self.action_std + self.action_mean

        action = action.cpu().numpy().astype(np.float32)
        action = np.clip(action, -1.0, 1.0)

        return action

    def _custom_reward(
        self,
        info,
        phase_before,
        phase_after,
        residual_action,
        terminated,
        truncated,
    ):
        reward = 0.0

        phase_id_before = self.phase_names.index(phase_before)
        phase_id_after = self.phase_names.index(phase_after)

        # 小的阶段推进奖励，不能太大，否则 PPO 会刷阶段奖励
        if phase_id_after > phase_id_before:
            reward += 1.0

        # 轻微鼓励抓起
        if info["cube_lifted"]:
            reward += 0.2

        # 残差惩罚：鼓励少改 BC
        reward -= 0.10 * float(np.linalg.norm(residual_action))

        # 每步小惩罚，鼓励快点完成
        reward -= 0.01

        # 最终奖励：只看完整 done 后是否成功
        if terminated:
            if info["is_success"]:
                reward += 100.0
            else:
                reward -= 80.0

        if truncated and not terminated:
            reward -= 30.0

        return float(reward)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0

        obs, info = self.base_env.reset(seed=seed)
        self.phase_controller.reset(info)

        self.last_obs = obs
        self.last_info = info

        aug_obs = self._get_aug_obs(obs)

        info = dict(info)
        info["phase"] = self.phase_controller.phase
        info["residual_scale"] = self.residual_scale

        return aug_obs, info

    def step(self, action):
        residual_action = np.asarray(action, dtype=np.float32)
        residual_action = np.clip(residual_action, -1.0, 1.0)

        phase_before = self.phase_controller.phase

        bc_action = self._bc_action(self.last_obs, phase_before)
        actual_action = np.clip(
            bc_action + self.residual_scale * residual_action,
            -1.0,
            1.0,
        ).astype(np.float32)

        obs, base_reward, base_terminated, base_truncated, info = self.base_env.step(actual_action)

        self.step_count += 1

        self.phase_controller.update(obs, info)
        phase_after = self.phase_controller.phase

        terminated = False
        truncated = False

        # 不因为 base_env 提前 terminated 就停。
        # 等完整 release / done。
        if phase_after == "done" and self.phase_controller.phase_steps >= 5:
            terminated = True

        if base_truncated or self.step_count >= self.max_episode_steps:
            truncated = True

        reward = self._custom_reward(
            info=info,
            phase_before=phase_before,
            phase_after=phase_after,
            residual_action=residual_action,
            terminated=terminated,
            truncated=truncated,
        )

        self.last_obs = obs
        self.last_info = info

        aug_obs = self._get_aug_obs(obs)

        info = dict(info)
        info["phase"] = phase_after
        info["phase_before"] = phase_before
        info["bc_action"] = bc_action
        info["residual_action"] = residual_action
        info["actual_action"] = actual_action
        info["base_reward"] = float(base_reward)
        info["base_terminated"] = bool(base_terminated)
        info["base_truncated"] = bool(base_truncated)
        info["wrapper_step_count"] = int(self.step_count)
        info["wrapper_terminated"] = bool(terminated)
        info["wrapper_truncated"] = bool(truncated)

        return aug_obs, reward, terminated, truncated, info

    def render(self):
        return self.base_env.render()

    def close(self):
        self.base_env.close()


def main():
    env = OpenArmPhaseResidualEnv(
        render_mode=None,
        residual_scale=0.05,
        randomize_cube=True,
        cube_x_range=(0.49, 0.55),
        cube_y_range=(-0.04, 0.04),
        max_episode_steps=260,
    )

    obs, info = env.reset(seed=123)

    print("环境 reset 成功")
    print("obs shape:", obs.shape)
    print("phase:", info["phase"])
    print("cube_pos:", info["cube_pos"])
    print("residual_scale:", info["residual_scale"])

    total_reward = 0.0

    # residual = 0，相当于纯 BC baseline
    for step in range(260):
        action = np.zeros(5, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 20 == 0:
            print(
                f"step={step:03d}, "
                f"phase={info['phase']}, "
                f"reward={reward:.3f}, "
                f"is_success={info['is_success']}, "
                f"terminated={terminated}, "
                f"truncated={truncated}"
            )

        if terminated or truncated:
            print(
                f"结束 step={step}, "
                f"phase={info['phase']}, "
                f"is_success={info['is_success']}, "
                f"terminated={terminated}, "
                f"truncated={truncated}"
            )
            break

    env.close()

    print("smoke test finished, total_reward:", total_reward)


if __name__ == "__main__":
    main()
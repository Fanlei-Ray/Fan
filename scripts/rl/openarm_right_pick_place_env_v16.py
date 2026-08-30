from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


@dataclass
class RightPickPlaceEnvConfig:
    xml_path: Optional[str] = None
    max_steps: int = 700
    settle_steps: int = 80
    frame_skip: int = 8
    randomize_cube: bool = True
    fixed_cube_pos: Tuple[float, float, float] = (0.516, 0.050, 1.050)
    cube_x_range: Tuple[float, float] = (0.49, 0.55)
    cube_y_range: Tuple[float, float] = (0.02, 0.08)
    cube_z: float = 1.050
    success_xy_threshold: float = 0.055
    success_z_margin: float = 0.005
    lift_success_delta_z: float = 0.020
    contact_penalty: float = 1.0
    dense_reward_scale: float = 1.0
    terminate_on_success: bool = True
    timeout_penalty: float = 12.0
    success_bonus: float = 120.0


class OpenArmRightPickPlaceEnv(gym.Env):
    """
    Gymnasium environment for right-arm OpenArm pick-and-place.

    This environment is intentionally state-based, not image-based. It is meant
    as the first RL integration point under the task planner. The upper-level
    planner can later call a trained policy through an adapter without changing
    planner logic.

    Action, normalized to [-1, 1]:
        0:7  right arm joint target delta
        7    right gripper command delta
        8    lifter command delta

    Observation:
        right_tcp_pos(3), cube_pos(3), frame_pos(3), tcp_to_cube_grasp(3),
        cube_to_frame(3), right_joint_qpos(7), right_ctrl(7),
        right_finger_ctrl(1), lifter_ctrl(1), cube_lift_delta(1),
        gripper_closed_hint(1), previous_pick_hint(1)
    """

    metadata = {"render_modes": [None, "human"], "render_fps": 60}

    RIGHT_ACTUATORS = [
        "right_joint1_ctrl",
        "right_joint2_ctrl",
        "right_joint3_ctrl",
        "right_joint4_ctrl",
        "right_joint5_ctrl",
        "right_joint6_ctrl",
        "right_joint7_ctrl",
        "right_finger1_ctrl",
        "lifter_ctrl",
    ]

    RIGHT_JOINT_ACTUATORS = [
        "right_joint1_ctrl",
        "right_joint2_ctrl",
        "right_joint3_ctrl",
        "right_joint4_ctrl",
        "right_joint5_ctrl",
        "right_joint6_ctrl",
        "right_joint7_ctrl",
    ]

    def __init__(self, config: Optional[RightPickPlaceEnvConfig] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.config = config or RightPickPlaceEnvConfig()
        self.render_mode = render_mode

        self.root = Path(__file__).resolve().parents[2]
        xml_path = self.config.xml_path
        if xml_path is None:
            xml_path = str(self.root / "v2" / "demo.xml")
        self.xml_path = Path(xml_path)
        if not self.xml_path.exists():
            raise FileNotFoundError(f"Cannot find MuJoCo XML: {self.xml_path}")

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        self.viewer = None
        self._step_count = 0
        self._initial_cube_z = 0.0
        self._ever_lifted = False
        self._last_dist_tcp_cube = None
        self._last_dist_cube_frame = None
        self._last_lift_delta = 0.0
        self._lift_bonus_given = False
        self._near_cube_bonus_given = False
        self._near_frame_bonus_given = False

        self._ids = self._build_ids()

        # Action scaling. Small deltas are more stable for PPO.
        self.action_scale = np.array([0.035] * 7 + [0.055, 0.018], dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)

        obs_dim = 3 + 3 + 3 + 3 + 3 + 7 + 7 + 1 + 1 + 1 + 1 + 1
        high = np.ones(obs_dim, dtype=np.float32) * np.inf
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Right-arm grasp heuristic target. Same successful rule-expert TCP offset.
        self.right_grasp_offset = np.array([-0.006, -0.020, 0.055], dtype=np.float64)

    def _build_ids(self) -> Dict[str, int]:
        ids = {}
        ids["right_tcp_site"] = self._site_id("right_gripper_tcp")
        ids["cube_body"] = self._body_id("orange_cube")
        ids["frame_body"] = self._body_id("black_frame")
        ids["home_key"] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        for name in self.RIGHT_ACTUATORS:
            ids[f"act_{name}"] = self._actuator_id(name)
        return ids

    def _body_id(self, name: str) -> int:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid == -1:
            raise ValueError(f"Missing body: {name}")
        return bid

    def _site_id(self, name: str) -> int:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid == -1:
            raise ValueError(f"Missing site: {name}")
        return sid

    def _actuator_id(self, name: str) -> int:
        aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid == -1:
            raise ValueError(f"Missing actuator: {name}")
        return aid

    def _sync_position_actuators_to_qpos(self) -> None:
        for aid in range(self.model.nu):
            jid = self.model.actuator_trnid[aid, 0]
            if jid < 0:
                continue
            qaddr = self.model.jnt_qposadr[jid]
            low, high = self.model.actuator_ctrlrange[aid]
            self.data.ctrl[aid] = np.clip(self.data.qpos[qaddr], low, high)

    def _set_ctrl(self, actuator_name: str, value: float) -> None:
        aid = self._ids[f"act_{actuator_name}"]
        low, high = self.model.actuator_ctrlrange[aid]
        self.data.ctrl[aid] = np.clip(float(value), low, high)

    def _get_ctrl(self, actuator_name: str) -> float:
        aid = self._ids[f"act_{actuator_name}"]
        return float(self.data.ctrl[aid])

    def _right_joint_qpos(self) -> np.ndarray:
        vals = []
        for act_name in self.RIGHT_JOINT_ACTUATORS:
            aid = self._ids[f"act_{act_name}"]
            jid = self.model.actuator_trnid[aid, 0]
            qaddr = self.model.jnt_qposadr[jid]
            vals.append(float(self.data.qpos[qaddr]))
        return np.asarray(vals, dtype=np.float32)

    def _right_joint_ctrl(self) -> np.ndarray:
        vals = []
        for act_name in self.RIGHT_JOINT_ACTUATORS:
            vals.append(self._get_ctrl(act_name))
        return np.asarray(vals, dtype=np.float32)

    def _set_free_body_pose(self, body_name: str, pos: np.ndarray) -> None:
        bid = self._body_id(body_name)
        if self.model.body_jntnum[bid] < 1:
            raise RuntimeError(f"Body {body_name} has no free joint")
        jid = self.model.body_jntadr[bid]
        qadr = self.model.jnt_qposadr[jid]
        dadr = self.model.jnt_dofadr[jid]
        self.data.qpos[qadr:qadr + 3] = np.asarray(pos, dtype=np.float64)
        self.data.qpos[qadr + 3:qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.data.qvel[dadr:dadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _sample_cube_pos(self) -> np.ndarray:
        if not self.config.randomize_cube:
            return np.asarray(self.config.fixed_cube_pos, dtype=np.float64)
        x = self.np_random.uniform(*self.config.cube_x_range)
        y = self.np_random.uniform(*self.config.cube_y_range)
        return np.asarray([x, y, self.config.cube_z], dtype=np.float64)

    def _get_tcp_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._ids["right_tcp_site"]].copy()

    def _get_cube_pos(self) -> np.ndarray:
        return self.data.xpos[self._ids["cube_body"]].copy()

    def _get_frame_pos(self) -> np.ndarray:
        return self.data.xpos[self._ids["frame_body"]].copy()

    def _get_obs(self) -> np.ndarray:
        tcp = self._get_tcp_pos()
        cube = self._get_cube_pos()
        frame = self._get_frame_pos()
        grasp_pos = cube + self.right_grasp_offset
        tcp_to_cube = grasp_pos - tcp
        cube_to_frame = frame - cube
        lift_delta = np.array([cube[2] - self._initial_cube_z], dtype=np.float32)
        finger = np.array([self._get_ctrl("right_finger1_ctrl")], dtype=np.float32)
        lifter = np.array([self._get_ctrl("lifter_ctrl")], dtype=np.float32)
        gripper_closed_hint = np.array([1.0 if finger[0] < 0.12 else 0.0], dtype=np.float32)
        ever_lifted = np.array([1.0 if self._ever_lifted else 0.0], dtype=np.float32)
        obs = np.concatenate(
            [
                tcp.astype(np.float32),
                cube.astype(np.float32),
                frame.astype(np.float32),
                tcp_to_cube.astype(np.float32),
                cube_to_frame.astype(np.float32),
                self._right_joint_qpos(),
                self._right_joint_ctrl(),
                finger,
                lifter,
                lift_delta,
                gripper_closed_hint,
                ever_lifted,
            ]
        )
        return obs.astype(np.float32)

    def _place_success(self) -> bool:
        cube = self._get_cube_pos()
        frame = self._get_frame_pos()
        xy_dist = float(np.linalg.norm(cube[:2] - frame[:2]))
        z_margin = float(cube[2] - frame[2])
        return bool(self._ever_lifted and xy_dist < self.config.success_xy_threshold and z_margin > self.config.success_z_margin)

    def _dangerous_contact_count(self) -> int:
        # Conservative filtered collision signal for RL penalty. Contact with task object is allowed.
        count = 0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
            g2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
            pair = f"{g1} {g2}".lower()
            if "orange_cube" in pair or "black_frame" in pair:
                continue
            if "cell_table_col" in pair and ("link4" in pair or "link5" in pair):
                continue
            if c.dist > -0.002:
                continue
            # penalize right arm hitting non-task world or left arm deeply
            if "right" in pair and ("left" in pair or "table" in pair or "ground" in pair or "world" in pair):
                count += 1
        return count

    def _compute_reward(self) -> Tuple[float, Dict[str, float | bool]]:
        """
        V16 reward fix.

        V15 had repeated per-step lift bonuses, so an episode could get a very
        large reward by lifting/holding without finishing place. V16 changes
        those into mostly one-time bonuses and bounded progress terms.
        """
        tcp = self._get_tcp_pos()
        cube = self._get_cube_pos()
        frame = self._get_frame_pos()
        grasp_pos = cube + self.right_grasp_offset

        dist_tcp_cube = float(np.linalg.norm(tcp - grasp_pos))
        dist_cube_frame = float(np.linalg.norm(cube[:2] - frame[:2]))
        lift_delta = float(cube[2] - self._initial_cube_z)

        if lift_delta > self.config.lift_success_delta_z:
            self._ever_lifted = True

        reward = 0.0

        # Time penalty: finish quickly, avoid 700-step timeout policies.
        reward -= 0.02

        # Reach the grasp pose.
        reward -= 1.25 * dist_tcp_cube
        if self._last_dist_tcp_cube is not None:
            progress = self._last_dist_tcp_cube - dist_tcp_cube
            reward += float(np.clip(6.0 * progress, -0.08, 0.08))
        self._last_dist_tcp_cube = dist_tcp_cube

        if dist_tcp_cube < 0.045:
            reward += 0.05
            if not self._near_cube_bonus_given:
                reward += 1.0
                self._near_cube_bonus_given = True

        # Lift progress and one-time lift success bonus.
        lift_progress = lift_delta - self._last_lift_delta
        reward += float(np.clip(8.0 * lift_progress, -0.12, 0.12))
        self._last_lift_delta = lift_delta

        if self._ever_lifted and not self._lift_bonus_given:
            reward += 12.0
            self._lift_bonus_given = True

        # After lifting, move cube to frame. Before lifting, do not reward
        # cube-frame distance too much, otherwise policy may push the cube.
        if self._ever_lifted:
            reward -= 1.6 * dist_cube_frame
            if self._last_dist_cube_frame is not None:
                frame_progress = self._last_dist_cube_frame - dist_cube_frame
                reward += float(np.clip(8.0 * frame_progress, -0.12, 0.12))
            if dist_cube_frame < 0.080:
                reward += 0.10
            if dist_cube_frame < self.config.success_xy_threshold and not self._near_frame_bonus_given:
                reward += 3.0
                self._near_frame_bonus_given = True
        else:
            # Tiny shaping only, so policy knows final target without preferring pushing.
            reward -= 0.05 * dist_cube_frame
        self._last_dist_cube_frame = dist_cube_frame

        success = self._place_success()
        if success:
            reward += self.config.success_bonus

        dangerous_contacts = self._dangerous_contact_count()
        if dangerous_contacts > 0:
            reward -= self.config.contact_penalty * dangerous_contacts

        # Dropping after lifting should hurt a lot.
        if self._ever_lifted and lift_delta < 0.005 and not success:
            reward -= 6.0

        info = {
            "dist_tcp_cube": dist_tcp_cube,
            "dist_cube_frame": dist_cube_frame,
            "lift_delta": lift_delta,
            "ever_lifted": self._ever_lifted,
            "place_success": success,
            "dangerous_contacts": dangerous_contacts,
            "lift_bonus_given": self._lift_bonus_given,
            "near_frame_bonus_given": self._near_frame_bonus_given,
        }
        return float(reward * self.config.dense_reward_scale), info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._step_count = 0
        self._ever_lifted = False
        self._last_dist_tcp_cube = None
        self._last_dist_cube_frame = None
        self._last_lift_delta = 0.0
        self._lift_bonus_given = False
        self._near_cube_bonus_given = False
        self._near_frame_bonus_given = False

        if self._ids["home_key"] != -1:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self._ids["home_key"])
        else:
            mujoco.mj_resetData(self.model, self.data)
        self._sync_position_actuators_to_qpos()

        cube_pos = self._sample_cube_pos()
        self._set_free_body_pose("orange_cube", cube_pos)
        self._set_ctrl("right_finger1_ctrl", 0.445)
        self._set_ctrl("lifter_ctrl", 0.0)
        mujoco.mj_forward(self.model, self.data)

        for _ in range(self.config.settle_steps):
            mujoco.mj_step(self.model, self.data)

        # Set cube exactly again after settling to avoid random drift at reset.
        self._set_free_body_pose("orange_cube", cube_pos)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

        self._initial_cube_z = float(self._get_cube_pos()[2])
        obs = self._get_obs()
        info = {
            "cube_pos": self._get_cube_pos(),
            "frame_pos": self._get_frame_pos(),
            "tcp_pos": self._get_tcp_pos(),
        }
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        # Apply normalized delta action to actuator controls.
        for idx, act_name in enumerate(self.RIGHT_ACTUATORS):
            aid = self._ids[f"act_{act_name}"]
            low, high = self.model.actuator_ctrlrange[aid]
            self.data.ctrl[aid] = np.clip(self.data.ctrl[aid] + action[idx] * self.action_scale[idx], low, high)

        for _ in range(self.config.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        reward, info = self._compute_reward()
        terminated = bool(info["place_success"] and self.config.terminate_on_success)
        truncated = bool(self._step_count >= self.config.max_steps)
        if truncated and not info["place_success"]:
            reward -= self.config.timeout_penalty
            info["timeout_penalty_applied"] = True
        else:
            info["timeout_penalty_applied"] = False
        obs = self._get_obs()
        info["step_count"] = self._step_count
        return obs, reward, terminated, truncated, info

    def render(self):
        # Training should be headless. Use test script for viewer rollouts.
        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

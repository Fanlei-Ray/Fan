from pathlib import Path
import numpy as np
import mujoco

import gymnasium as gym
from gymnasium import spaces

# 复用你已经跑通的 right_pick_place.py 里的配置和工具
import right_pick_place as pp


class OpenArmPickPlaceEnv(gym.Env):
    """
    OpenArm MuJoCo pick-and-place Gymnasium 环境 v0

    action: shape=(5,)
        action[0] = TCP x 方向增量
        action[1] = TCP y 方向增量
        action[2] = TCP z 方向增量
        action[3] = 夹爪开合，正数张开，负数闭合
        action[4] = lifter 升降，正数上升，负数下降

    observation:
        tcp_pos                 3
        cube_pos                3
        frame_pos               3
        cube_pos - tcp_pos      3
        frame_pos - cube_pos    3
        left_arm_qpos           7
        finger_ctrl             1
        lifter_ctrl             1
        total                  24
    """

    metadata = {
        "render_modes": ["human"],
        "render_fps": 50,
    }

    def __init__(
        self,
        render_mode=None,
        randomize_cube=True,
        cube_x_range=(0.49, 0.55),
        cube_y_range=(-0.04, 0.04),
        cube_z=1.05,
        max_episode_steps=150,
        frame_skip=20,
    ):
        super().__init__()

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"不支持的 render_mode: {render_mode}")

        self.render_mode = render_mode
        self.viewer = None

        self.randomize_cube = randomize_cube
        self.cube_x_range = cube_x_range
        self.cube_y_range = cube_y_range
        self.cube_z = cube_z

        self.max_episode_steps = max_episode_steps
        self.frame_skip = frame_skip
        self.elapsed_steps = 0

        self.model = mujoco.MjModel.from_xml_path(str(pp.XML_PATH))
        self.data = mujoco.MjData(self.model)

        # 关键对象 id
        self.tcp_sid = pp.site_id(self.model, pp.LEFT_SITE_NAME)
        self.cube_bid = pp.body_id(self.model, pp.PICK_BODY_NAME)
        self.frame_bid = pp.body_id(self.model, pp.PLACE_BODY_NAME)

        self.finger_aid = pp.actuator_id(self.model, "left_finger1_ctrl")
        self.lifter_aid = pp.actuator_id(self.model, "lifter_ctrl")

        self.left_actuator_ids = [
            pp.actuator_id(self.model, name)
            for name in pp.LEFT_ACTUATOR_NAMES
        ]

        self.left_qpos_addrs = []
        self.left_dof_addrs = []
        self.left_joint_ranges = []

        for name in pp.LEFT_JOINT_NAMES:
            jid = pp.joint_id(self.model, name)
            self.left_qpos_addrs.append(self.model.jnt_qposadr[jid])
            self.left_dof_addrs.append(self.model.jnt_dofadr[jid])
            self.left_joint_ranges.append(self.model.jnt_range[jid].copy())

        self.left_joint_ranges = np.array(self.left_joint_ranges)

        # action: dx, dy, dz, gripper, lifter
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(24,),
            dtype=np.float32,
        )

        # 每个 step 的最大物理变化量
        self.max_tcp_delta = 0.02       # 2 cm
        self.max_finger_delta = 0.08
        self.max_lifter_delta = 0.02    # 2 cm

        self.initial_cube_z = None

        self._load_home_silent()

    # ============================================================
    # 基础状态
    # ============================================================

    def _load_home_silent(self):
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")

        if key_id != -1:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)

        # 同步 position actuator ctrl
        for aid in range(self.model.nu):
            jid = self.model.actuator_trnid[aid, 0]
            if jid < 0:
                continue

            qaddr = self.model.jnt_qposadr[jid]
            low, high = self.model.actuator_ctrlrange[aid]
            self.data.ctrl[aid] = np.clip(self.data.qpos[qaddr], low, high)

        mujoco.mj_forward(self.model, self.data)

    def _find_freejoint_for_body(self, body_name):
        bid = pp.body_id(self.model, body_name)

        for jid in range(self.model.njnt):
            if self.model.jnt_bodyid[jid] != bid:
                continue

            if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                return jid

        raise ValueError(f"{body_name} 没有找到 freejoint")

    def _set_free_body_pose(self, body_name, pos, quat=None):
        if quat is None:
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

        jid = self._find_freejoint_for_body(body_name)

        qaddr = self.model.jnt_qposadr[jid]
        daddr = self.model.jnt_dofadr[jid]

        self.data.qpos[qaddr:qaddr + 3] = np.asarray(pos, dtype=float)
        self.data.qpos[qaddr + 3:qaddr + 7] = np.asarray(quat, dtype=float)

        self.data.qvel[daddr:daddr + 6] = 0.0

        mujoco.mj_forward(self.model, self.data)

    def _sample_cube_pos(self):
        if not self.randomize_cube:
            return np.array([0.52, 0.0, self.cube_z], dtype=float)

        x = self.np_random.uniform(self.cube_x_range[0], self.cube_x_range[1])
        y = self.np_random.uniform(self.cube_y_range[0], self.cube_y_range[1])
        z = self.cube_z

        return np.array([x, y, z], dtype=float)

    def _step_sim(self, n_steps):
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

        if self.render_mode == "human":
            self.render()

    # ============================================================
    # Observation / reward / done
    # ============================================================

    def _get_obs(self):
        tcp_pos = self.data.site_xpos[self.tcp_sid].copy()
        cube_pos = self.data.xpos[self.cube_bid].copy()
        frame_pos = self.data.xpos[self.frame_bid].copy()

        left_qpos = np.array(
            [self.data.qpos[qaddr] for qaddr in self.left_qpos_addrs],
            dtype=float,
        )

        finger_ctrl = np.array([self.data.ctrl[self.finger_aid]], dtype=float)
        lifter_ctrl = np.array([self.data.ctrl[self.lifter_aid]], dtype=float)

        obs = np.concatenate(
            [
                tcp_pos,
                cube_pos,
                frame_pos,
                cube_pos - tcp_pos,
                frame_pos - cube_pos,
                left_qpos,
                finger_ctrl,
                lifter_ctrl,
            ]
        ).astype(np.float32)

        return obs

    def _get_info(self):
        tcp_pos = self.data.site_xpos[self.tcp_sid].copy()
        cube_pos = self.data.xpos[self.cube_bid].copy()
        frame_pos = self.data.xpos[self.frame_bid].copy()

        grasp_target = cube_pos + pp.PICK_GRASP_OFFSET

        dist_tcp_to_grasp = float(np.linalg.norm(tcp_pos - grasp_target))
        dist_cube_to_frame_xy = float(np.linalg.norm(cube_pos[:2] - frame_pos[:2]))

        cube_lifted = False
        if self.initial_cube_z is not None:
            cube_lifted = cube_pos[2] > self.initial_cube_z + pp.LIFT_SUCCESS_DELTA_Z

        is_success = self._is_success()

        return {
            "tcp_pos": tcp_pos,
            "cube_pos": cube_pos,
            "frame_pos": frame_pos,
            "dist_tcp_to_grasp": dist_tcp_to_grasp,
            "dist_cube_to_frame_xy": dist_cube_to_frame_xy,
            "cube_lifted": bool(cube_lifted),
            "is_success": bool(is_success),
        }

    def _is_success(self):
        cube_pos = self.data.xpos[self.cube_bid].copy()
        frame_pos = self.data.xpos[self.frame_bid].copy()

        xy_close = np.linalg.norm(cube_pos[:2] - frame_pos[:2]) < 0.045
        z_ok = cube_pos[2] > frame_pos[2] + 0.015

        return bool(xy_close and z_ok)

    def _compute_reward(self, action, ik_error, ik_success):
        tcp_pos = self.data.site_xpos[self.tcp_sid].copy()
        cube_pos = self.data.xpos[self.cube_bid].copy()
        frame_pos = self.data.xpos[self.frame_bid].copy()

        grasp_target = cube_pos + pp.PICK_GRASP_OFFSET

        dist_tcp_to_grasp = np.linalg.norm(tcp_pos - grasp_target)
        dist_cube_to_frame_xy = np.linalg.norm(cube_pos[:2] - frame_pos[:2])

        reward = 0.0

        # 先鼓励 TCP 靠近方块抓取点
        reward += -2.0 * dist_tcp_to_grasp

        # 抓起来以后，鼓励方块靠近黑框
        cube_lifted = cube_pos[2] > self.initial_cube_z + pp.LIFT_SUCCESS_DELTA_Z

        if cube_lifted:
            reward += 3.0
            reward += -2.0 * dist_cube_to_frame_xy

        # 成功放到黑框附近
        if self._is_success():
            reward += 20.0

        # 动作别太大
        reward += -0.01 * float(np.linalg.norm(action))

        # IK 太差时惩罚
        if not ik_success:
            reward += -min(2.0, 10.0 * ik_error)

        return float(reward)

    # ============================================================
    # IK
    # ============================================================

    def _make_ik_data(self):
        ik_data = mujoco.MjData(self.model)

        ik_data.qpos[:] = self.data.qpos[:]
        ik_data.qvel[:] = self.data.qvel[:]
        ik_data.ctrl[:] = self.data.ctrl[:]

        mujoco.mj_forward(self.model, ik_data)
        return ik_data

    def _solve_ik_silent(
        self,
        target_pos,
        max_iters=120,
        tolerance=1e-3,
        step_size=0.6,
        damping=1e-3,
    ):
        ik_data = self._make_ik_data()
        target_pos = np.asarray(target_pos, dtype=float)

        for _ in range(max_iters):
            mujoco.mj_forward(self.model, ik_data)

            current_pos = ik_data.site_xpos[self.tcp_sid].copy()
            error = target_pos - current_pos
            err_norm = np.linalg.norm(error)

            if err_norm < tolerance:
                break

            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, ik_data, jacp, jacr, self.tcp_sid)

            J = jacp[:, self.left_dof_addrs]

            A = J @ J.T + damping * np.eye(3)
            dq = J.T @ np.linalg.solve(A, error)

            dq = step_size * dq
            dq = np.clip(dq, -0.05, 0.05)

            for i, qaddr in enumerate(self.left_qpos_addrs):
                ik_data.qpos[qaddr] += dq[i]

                low, high = self.left_joint_ranges[i]
                ik_data.qpos[qaddr] = np.clip(ik_data.qpos[qaddr], low, high)

        mujoco.mj_forward(self.model, ik_data)

        final_pos = ik_data.site_xpos[self.tcp_sid].copy()
        final_error = float(np.linalg.norm(target_pos - final_pos))
        success = final_error < 0.02

        ctrl_targets = []

        for aid, qaddr in zip(self.left_actuator_ids, self.left_qpos_addrs):
            low, high = self.model.actuator_ctrlrange[aid]
            value = float(np.clip(ik_data.qpos[qaddr], low, high))
            ctrl_targets.append(value)

        return success, final_error, np.array(ctrl_targets, dtype=float)

    # ============================================================
    # Gymnasium API
    # ============================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.elapsed_steps = 0

        self._load_home_silent()

        cube_pos = self._sample_cube_pos()
        self._set_free_body_pose(pp.PICK_BODY_NAME, cube_pos)

        # 让方块落稳
        self._step_sim(200)

        self.initial_cube_z = float(self.data.xpos[self.cube_bid][2])

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.elapsed_steps += 1

        tcp_pos = self.data.site_xpos[self.tcp_sid].copy()
        target_tcp_pos = tcp_pos + action[:3] * self.max_tcp_delta

        ik_success, ik_error, left_ctrl_targets = self._solve_ik_silent(target_tcp_pos)

        # 应用左臂 IK ctrl
        for aid, value in zip(self.left_actuator_ids, left_ctrl_targets):
            low, high = self.model.actuator_ctrlrange[aid]
            self.data.ctrl[aid] = np.clip(value, low, high)

        # 夹爪控制：正数张开，负数闭合
        finger_low, finger_high = self.model.actuator_ctrlrange[self.finger_aid]
        self.data.ctrl[self.finger_aid] = np.clip(
            self.data.ctrl[self.finger_aid] + action[3] * self.max_finger_delta,
            finger_low,
            finger_high,
        )

        # lifter 控制
        lifter_low, lifter_high = self.model.actuator_ctrlrange[self.lifter_aid]
        self.data.ctrl[self.lifter_aid] = np.clip(
            self.data.ctrl[self.lifter_aid] + action[4] * self.max_lifter_delta,
            lifter_low,
            lifter_high,
        )

        self._step_sim(self.frame_skip)

        obs = self._get_obs()
        reward = self._compute_reward(action, ik_error, ik_success)

        terminated = self._is_success()
        truncated = self.elapsed_steps >= self.max_episode_steps

        info = self._get_info()
        info["ik_success"] = bool(ik_success)
        info["ik_error"] = float(ik_error)
        info["elapsed_steps"] = int(self.elapsed_steps)

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return

        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
from pathlib import Path
import sys
import csv
import time

import mujoco
import numpy as np
import torch
import torch.nn as nn


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import right_rule_pick_place as rr


# ============================================================
# Test right-arm phase-aware BC policy v1
#
# 输入:
#   outputs/right_bc_v1/right_bc_policy_phase_v1.pt
#
# 输出:
#   outputs/right_bc_v1/right_bc_test_v1_log.csv
# ============================================================


RESULT_DIR = ROOT / "outputs" / "right_bc_v1"
MODEL_PATH = RESULT_DIR / "right_bc_policy_phase_v1.pt"
LOG_PATH = RESULT_DIR / "right_bc_test_v1_log.csv"

NUM_EPISODES = 30
SEED_START = 50001

CUBE_X_RANGE = (0.49, 0.55)
CUBE_Y_RANGE = (0.02, 0.08)
CUBE_Z = 1.050

OBS_DIM = 24
ACTION_DIM = 5

CONTROL_STEPS = 40
MAX_DELTA = 0.060

# 初版测试：BC 负责 TCP 运动，phase 负责开合夹爪和 lifter。
# 这样先验证最关键的右臂运动 BC。
USE_POLICY_GRIPPER_LIFTER = False

# 是否打印每个 control tick 的 debug，默认关。
VERBOSE_STEP = False


BEST_CONFIG = {
    "name": "right_best_tcp_x-0.006_y-0.020_z+0.055__j7_m060",
    "site_type": "tcp",

    "pregrasp_offset": np.array([-0.006, -0.020, 0.140], dtype=float),
    "grasp_offset": np.array([-0.006, -0.020, 0.055], dtype=float),

    "preplace_offset": np.array([0.0, 0.0, 0.140], dtype=float),
    "place_offset": np.array([0.0, 0.0, 0.080], dtype=float),

    "joint_biases": {
        "right_joint7_ctrl": -0.060,
    },
}


PHASE_PREGRASP = 0
PHASE_GRASP = 1
PHASE_CLOSE = 2
PHASE_LIFT = 3
PHASE_PREPLACE = 4
PHASE_PLACE = 5
PHASE_RELEASE = 6

NUM_PHASES = 7

PHASE_NAMES = {
    PHASE_PREGRASP: "pregrasp",
    PHASE_GRASP: "grasp",
    PHASE_CLOSE: "close",
    PHASE_LIFT: "lift",
    PHASE_PREPLACE: "preplace",
    PHASE_PLACE: "place",
    PHASE_RELEASE: "release",
}


class RightBCPolicy(nn.Module):
    def __init__(self, obs_dim=24, action_dim=5):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),

            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),

            nn.Linear(128, 128),
            nn.SiLU(),

            nn.Linear(128, action_dim),
        )

    def forward(self, obs):
        return self.net(obs)


def sample_cube_pos(seed):
    rng = np.random.default_rng(seed)

    x = rng.uniform(CUBE_X_RANGE[0], CUBE_X_RANGE[1])
    y = rng.uniform(CUBE_Y_RANGE[0], CUBE_Y_RANGE[1])

    return np.array([x, y, CUBE_Z], dtype=float)


def actuator_ctrl(model, data, name):
    aid = rr.actuator_id(model, name)
    return float(data.ctrl[aid])


def right_joint_qpos(model, data):
    values = []

    for joint_name in rr.RIGHT_JOINT_NAMES:
        jid = rr.joint_id(model, joint_name)
        qaddr = model.jnt_qposadr[jid]
        values.append(float(data.qpos[qaddr]))

    return np.array(values, dtype=np.float32)


def right_gripper_norm(model, data):
    ctrl = actuator_ctrl(model, data, "right_finger1_ctrl")

    denom = abs(rr.RIGHT_FINGER_PRE_OPEN - rr.RIGHT_FINGER_CLOSE)
    if denom < 1e-6:
        return 0.0

    value = (rr.RIGHT_FINGER_CLOSE - ctrl) / denom
    return float(np.clip(value, 0.0, 1.0))


def make_obs(model, data, site_name, phase_id):
    site_pos = rr.get_site_pos(model, data, site_name).astype(np.float32)
    cube_pos = rr.get_body_pos(model, data, "orange_cube").astype(np.float32)
    frame_pos = rr.get_body_pos(model, data, "black_frame").astype(np.float32)

    cube_rel_site = cube_pos - site_pos
    frame_rel_cube = frame_pos - cube_pos

    joints = right_joint_qpos(model, data)

    grip = np.array([right_gripper_norm(model, data)], dtype=np.float32)
    phase = np.array([float(phase_id) / float(NUM_PHASES - 1)], dtype=np.float32)

    obs = np.concatenate(
        [
            site_pos,          # 3
            cube_pos,          # 3
            frame_pos,         # 3
            cube_rel_site,     # 3
            frame_rel_cube,    # 3
            joints,            # 7
            grip,              # 1
            phase,             # 1
        ],
        axis=0,
    ).astype(np.float32)

    assert obs.shape == (24,), obs.shape
    return obs


def load_policy(device):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型：{MODEL_PATH}")

    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    model = RightBCPolicy(
        obs_dim=int(ckpt["obs_dim"]),
        action_dim=int(ckpt["action_dim"]),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    obs_mean = np.asarray(ckpt["obs_mean"], dtype=np.float32)
    obs_std = np.asarray(ckpt["obs_std"], dtype=np.float32)
    action_mean = np.asarray(ckpt["action_mean"], dtype=np.float32)
    action_std = np.asarray(ckpt["action_std"], dtype=np.float32)

    print("=" * 80)
    print("Loaded right BC policy")
    print("=" * 80)
    print("MODEL_PATH:", MODEL_PATH)
    print("best_epoch:", ckpt["best_epoch"])
    print("best_val_loss:", ckpt["best_val_loss"])
    print("val_mae:", ckpt["val_mae"])
    print("val_rmse:", ckpt["val_rmse"])
    print("=" * 80)

    return model, obs_mean, obs_std, action_mean, action_std


def policy_action(policy, obs, obs_mean, obs_std, action_mean, action_std, device):
    obs_norm = ((obs - obs_mean) / obs_std).astype(np.float32)

    with torch.no_grad():
        x = torch.from_numpy(obs_norm).float().unsqueeze(0).to(device)
        pred_norm = policy(x).squeeze(0).detach().cpu().numpy()

    action = pred_norm * action_std + action_mean

    action = action.astype(np.float32)
    action[:3] = np.clip(action[:3], -MAX_DELTA, MAX_DELTA)
    action[3] = float(np.clip(action[3], 0.0, 1.0))
    action[4] = float(np.clip(action[4], 0.0, 1.0))

    return action


def load_home_quiet(model, data):
    key_id = rr.maybe_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)

    rr.sync_position_actuators_to_qpos(model, data)
    mujoco.mj_forward(model, data)


def reset_scene_quiet(model, data, cube_pos):
    load_home_quiet(model, data)

    rr.set_free_body_pos(model, data, "orange_cube", cube_pos)

    rr.set_ctrl(model, data, "right_finger1_ctrl", rr.RIGHT_FINGER_PRE_OPEN)
    rr.set_ctrl(model, data, "lifter_ctrl", 0.0)

    mujoco.mj_forward(model, data)

    rr.sim_steps(model, data, steps=700, viewer=None, realtime=False)

    rr.set_free_body_pos(model, data, "orange_cube", cube_pos)

    rr.set_ctrl(model, data, "right_finger1_ctrl", rr.RIGHT_FINGER_PRE_OPEN)
    rr.set_ctrl(model, data, "lifter_ctrl", 0.0)

    mujoco.mj_forward(model, data)

    rr.sim_steps(model, data, steps=180, viewer=None, realtime=False)


def phase_target(model, data, site_name, phase_id):
    if phase_id == PHASE_PREGRASP:
        cube_pos = rr.get_body_pos(model, data, "orange_cube")
        return cube_pos + BEST_CONFIG["pregrasp_offset"]

    if phase_id == PHASE_GRASP:
        cube_pos = rr.get_body_pos(model, data, "orange_cube")
        return cube_pos + BEST_CONFIG["grasp_offset"]

    if phase_id == PHASE_PREPLACE:
        frame_pos = rr.get_body_pos(model, data, "black_frame")
        return frame_pos + BEST_CONFIG["preplace_offset"]

    if phase_id == PHASE_PLACE:
        frame_pos = rr.get_body_pos(model, data, "black_frame")
        return frame_pos + BEST_CONFIG["place_offset"]

    return rr.get_site_pos(model, data, site_name)


def scheduled_gripper_lifter(phase_id):
    if phase_id in [PHASE_PREGRASP, PHASE_GRASP]:
        return 1.0, 0.0

    if phase_id == PHASE_CLOSE:
        return 0.0, 0.0

    if phase_id in [PHASE_LIFT, PHASE_PREPLACE, PHASE_PLACE]:
        return 0.0, 1.0

    if phase_id == PHASE_RELEASE:
        return 1.0, 1.0

    return 1.0, 0.0


def set_gripper_and_lifter(model, data, phase_id, gripper_cmd, lifter_cmd):
    if phase_id == PHASE_RELEASE:
        open_value = rr.RIGHT_FINGER_OPEN
    else:
        open_value = rr.RIGHT_FINGER_PRE_OPEN

    if gripper_cmd >= 0.5:
        finger_value = open_value
    else:
        finger_value = rr.RIGHT_FINGER_CLOSE

    if lifter_cmd >= 0.5:
        lifter_value = rr.LIFTER_UP
    else:
        lifter_value = 0.0

    rr.set_ctrl(model, data, "right_finger1_ctrl", finger_value)
    rr.set_ctrl(model, data, "lifter_ctrl", lifter_value)


def apply_tcp_delta_control(model, data, site_name, delta_xyz):
    delta_xyz = np.asarray(delta_xyz, dtype=float)
    delta_xyz = np.clip(delta_xyz, -MAX_DELTA, MAX_DELTA)

    sid = rr.site_id(model, site_name)

    qpos_addrs = []
    dof_addrs = []
    actuator_ids = []

    for joint_name, actuator_name in zip(rr.RIGHT_JOINT_NAMES, rr.RIGHT_ACTUATOR_NAMES):
        jid = rr.joint_id(model, joint_name)
        aid = rr.actuator_id(model, actuator_name)

        qpos_addrs.append(model.jnt_qposadr[jid])
        dof_addrs.append(model.jnt_dofadr[jid])
        actuator_ids.append(aid)

    mujoco.mj_forward(model, data)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, sid)

    J = jacp[:, dof_addrs]

    damping = 3e-3
    A = J @ J.T + damping * np.eye(3)

    try:
        dq = J.T @ np.linalg.solve(A, delta_xyz)
    except np.linalg.LinAlgError:
        dq = np.zeros(len(dof_addrs), dtype=float)

    dq = 0.65 * dq
    dq = np.clip(dq, -0.035, 0.035)

    for i, aid in enumerate(actuator_ids):
        qaddr = qpos_addrs[i]
        low, high = model.actuator_ctrlrange[aid]

        target = float(data.qpos[qaddr] + dq[i])
        data.ctrl[aid] = np.clip(target, low, high)


def sim_control_steps(model, data, steps):
    max_z = -1e9

    for _ in range(steps):
        mujoco.mj_step(model, data)
        cube_pos = rr.get_body_pos(model, data, "orange_cube")
        max_z = max(max_z, float(cube_pos[2]))

    return max_z


def should_advance_phase(model, data, site_name, phase_id, phase_tick, cube_initial_z):
    site_pos = rr.get_site_pos(model, data, site_name)
    cube_pos = rr.get_body_pos(model, data, "orange_cube")
    frame_pos = rr.get_body_pos(model, data, "black_frame")

    target = phase_target(model, data, site_name, phase_id)
    dist = float(np.linalg.norm(site_pos - target))

    cube_lift = float(cube_pos[2] - cube_initial_z)

    if phase_id == PHASE_PREGRASP:
        if phase_tick >= 18 and dist < 0.040:
            return True
        if phase_tick >= 65:
            return True
        return False

    if phase_id == PHASE_GRASP:
        if phase_tick >= 16 and dist < 0.032:
            return True
        if phase_tick >= 55:
            return True
        return False

    if phase_id == PHASE_CLOSE:
        if phase_tick >= 65:
            return True
        return False

    if phase_id == PHASE_LIFT:
        if phase_tick >= 24 and cube_lift > 0.060:
            return True
        if phase_tick >= 65:
            return True
        return False

    if phase_id == PHASE_PREPLACE:
        if phase_tick >= 18 and dist < 0.055:
            return True
        if phase_tick >= 70:
            return True
        return False

    if phase_id == PHASE_PLACE:
        if phase_tick >= 16 and dist < 0.045:
            return True
        if phase_tick >= 60:
            return True
        return False

    if phase_id == PHASE_RELEASE:
        if phase_tick >= 60:
            return True
        return False

    return False


def run_episode(policy, obs_mean, obs_std, action_mean, action_std, device, model, site_name, seed):
    data = mujoco.MjData(model)

    cube_pos = sample_cube_pos(seed)
    reset_scene_quiet(model, data, cube_pos)

    cube_initial = rr.get_body_pos(model, data, "orange_cube")
    cube_initial_z = float(cube_initial[2])

    phase_id = PHASE_PREGRASP
    phase_tick = 0
    total_tick = 0

    max_cube_z = cube_initial_z

    last_action = np.zeros(5, dtype=np.float32)

    while True:
        total_tick += 1
        phase_tick += 1

        obs = make_obs(model, data, site_name, phase_id)

        action = policy_action(
            policy=policy,
            obs=obs,
            obs_mean=obs_mean,
            obs_std=obs_std,
            action_mean=action_mean,
            action_std=action_std,
            device=device,
        )

        last_action = action.copy()

        if USE_POLICY_GRIPPER_LIFTER:
            gripper_cmd = float(action[3])
            lifter_cmd = float(action[4])
        else:
            gripper_cmd, lifter_cmd = scheduled_gripper_lifter(phase_id)

        apply_tcp_delta_control(
            model=model,
            data=data,
            site_name=site_name,
            delta_xyz=action[:3],
        )

        set_gripper_and_lifter(
            model=model,
            data=data,
            phase_id=phase_id,
            gripper_cmd=gripper_cmd,
            lifter_cmd=lifter_cmd,
        )

        tick_max_z = sim_control_steps(model, data, CONTROL_STEPS)
        max_cube_z = max(max_cube_z, tick_max_z)

        if VERBOSE_STEP:
            site_pos = rr.get_site_pos(model, data, site_name)
            target = phase_target(model, data, site_name, phase_id)
            dist = float(np.linalg.norm(site_pos - target))
            print(
                f"tick={total_tick:04d} "
                f"phase={PHASE_NAMES[phase_id]} "
                f"phase_tick={phase_tick:03d} "
                f"dist={dist:.4f} "
                f"action={np.array2string(action, precision=4)}"
            )

        advance = should_advance_phase(
            model=model,
            data=data,
            site_name=site_name,
            phase_id=phase_id,
            phase_tick=phase_tick,
            cube_initial_z=cube_initial_z,
        )

        if advance:
            if phase_id == PHASE_RELEASE:
                break

            phase_id += 1
            phase_tick = 0

        if total_tick >= 460:
            break

    final_cube = rr.get_body_pos(model, data, "orange_cube")
    frame_pos = rr.get_body_pos(model, data, "black_frame")

    lift_delta = float(max_cube_z - cube_initial_z)
    final_lift_delta = float(final_cube[2] - cube_initial_z)

    pick_success = lift_delta > rr.LIFT_SUCCESS_DELTA_Z

    xy_dist = float(np.linalg.norm(final_cube[:2] - frame_pos[:2]))
    z_margin = float(final_cube[2] - frame_pos[2])

    place_success = bool(pick_success and xy_dist < 0.055 and z_margin > 0.005)

    if place_success:
        fail_reason = "success"
    elif not pick_success:
        fail_reason = "pick_lift_fail"
    elif xy_dist >= 0.055:
        fail_reason = "place_xy_fail"
    elif z_margin <= 0.005:
        fail_reason = "place_z_fail"
    else:
        fail_reason = "unknown_fail"

    return {
        "seed": int(seed),
        "cube_initial": cube_initial.copy(),
        "cube_final": final_cube.copy(),
        "frame_final": frame_pos.copy(),
        "pick_success": bool(pick_success),
        "place_success": bool(place_success),
        "lift_delta": float(lift_delta),
        "final_lift_delta": float(final_lift_delta),
        "xy_dist": float(xy_dist),
        "z_margin": float(z_margin),
        "fail_reason": fail_reason,
        "total_tick": int(total_tick),
        "last_phase": PHASE_NAMES.get(phase_id, str(phase_id)),
        "last_action": last_action.copy(),
    }


def init_log():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "episode",
                "seed",
                "cube_x",
                "cube_y",
                "pick_success",
                "place_success",
                "lift_delta",
                "final_lift_delta",
                "xy_dist",
                "z_margin",
                "fail_reason",
                "total_tick",
                "last_phase",
            ]
        )


def append_log(ep, result):
    cube = result["cube_initial"]

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                ep,
                result["seed"],
                float(cube[0]),
                float(cube[1]),
                bool(result["pick_success"]),
                bool(result["place_success"]),
                float(result["lift_delta"]),
                float(result["final_lift_delta"]),
                float(result["xy_dist"]),
                float(result["z_margin"]),
                result["fail_reason"],
                int(result["total_tick"]),
                result["last_phase"],
            ]
        )


def main():
    print("=" * 80)
    print("Test right-arm BC phase policy v1")
    print("=" * 80)
    print("ROOT:", ROOT)
    print("MODEL_PATH:", MODEL_PATH)
    print("LOG_PATH:", LOG_PATH)
    print("NUM_EPISODES:", NUM_EPISODES)
    print("CUBE_X_RANGE:", CUBE_X_RANGE)
    print("CUBE_Y_RANGE:", CUBE_Y_RANGE)
    print("USE_POLICY_GRIPPER_LIFTER:", USE_POLICY_GRIPPER_LIFTER)
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if torch.cuda.is_available():
        print("cuda:", torch.cuda.get_device_name(0))

    policy, obs_mean, obs_std, action_mean, action_std = load_policy(device)

    model = mujoco.MjModel.from_xml_path(str(rr.XML_PATH))
    site_name = rr.choose_right_site(model)

    print("使用右臂 site:", site_name)

    init_log()

    results = []

    start_time = time.time()

    for ep in range(1, NUM_EPISODES + 1):
        seed = SEED_START + ep

        result = run_episode(
            policy=policy,
            obs_mean=obs_mean,
            obs_std=obs_std,
            action_mean=action_mean,
            action_std=action_std,
            device=device,
            model=model,
            site_name=site_name,
            seed=seed,
        )

        results.append(result)
        append_log(ep, result)

        mark = "O" if result["place_success"] else ("P" if result["pick_success"] else "X")
        cube = result["cube_initial"]

        print(
            f"{mark} ep={ep:02d}/{NUM_EPISODES} "
            f"seed={seed} "
            f"cube=({cube[0]:.3f},{cube[1]:.3f}) "
            f"pick={result['pick_success']} "
            f"place={result['place_success']} "
            f"lift={result['lift_delta']:.4f} "
            f"final_lift={result['final_lift_delta']:.4f} "
            f"xy={result['xy_dist']:.4f} "
            f"z_margin={result['z_margin']:.4f} "
            f"reason={result['fail_reason']} "
            f"ticks={result['total_tick']}"
        )

    elapsed = time.time() - start_time

    pick_count = sum(1 for r in results if r["pick_success"])
    place_count = sum(1 for r in results if r["place_success"])

    reason_counts = {}
    for r in results:
        reason = r["fail_reason"]
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    print("")
    print("=" * 80)
    print("Right BC policy test 总结")
    print("=" * 80)
    print(f"pick_success：{pick_count}/{NUM_EPISODES}")
    print(f"pick_success 成功率：{pick_count / NUM_EPISODES * 100:.1f}%")
    print(f"place_success：{place_count}/{NUM_EPISODES}")
    print(f"place_success 成功率：{place_count / NUM_EPISODES * 100:.1f}%")
    print("reason_counts:", reason_counts)
    print("耗时:", f"{elapsed / 60:.1f} min")
    print("LOG_PATH:", LOG_PATH)

    print("")
    print("失败样本：")
    for r in results:
        if not r["place_success"]:
            cube = r["cube_initial"]
            print(
                f"seed={r['seed']} "
                f"cube=({cube[0]:.3f},{cube[1]:.3f}) "
                f"pick={r['pick_success']} "
                f"place={r['place_success']} "
                f"lift={r['lift_delta']:.4f} "
                f"xy={r['xy_dist']:.4f} "
                f"z_margin={r['z_margin']:.4f} "
                f"reason={r['fail_reason']} "
                f"last_phase={r['last_phase']}"
            )


if __name__ == "__main__":
    main()
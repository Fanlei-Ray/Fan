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
# Test right-arm DIRECT actuator BC policy v1
#
# 输入:
#   outputs/right_bc_v1/right_bc_direct_policy_v1.pt
#
# 输出:
#   outputs/right_bc_v1/right_bc_direct_test_v1_log.csv
#
# 关键：
#   policy 输出 9 维 actuator ctrl
#   直接写入 data.ctrl
#   不再使用 Jacobian
# ============================================================


RESULT_DIR = ROOT / "outputs" / "right_bc_v1"
MODEL_PATH = RESULT_DIR / "right_bc_direct_policy_v1.pt"
LOG_PATH = RESULT_DIR / "right_bc_direct_test_v1_log.csv"

NUM_EPISODES = 30
SEED_START = 60001

CUBE_X_RANGE = (0.49, 0.55)
CUBE_Y_RANGE = (0.02, 0.08)
CUBE_Z = 1.050

OBS_DIM = 24
ACTION_DIM = 9

CONTROL_STEPS = 40

VERBOSE_STEP = False


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


DEFAULT_ACTION_ACTUATORS = [
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


class RightDirectBCPolicy(nn.Module):
    def __init__(self, obs_dim=24, action_dim=9):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),

            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),

            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
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
            site_pos,
            cube_pos,
            frame_pos,
            cube_rel_site,
            frame_rel_cube,
            joints,
            grip,
            phase,
        ],
        axis=0,
    ).astype(np.float32)

    assert obs.shape == (OBS_DIM,), obs.shape
    return obs


def load_policy(device):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型：{MODEL_PATH}")

    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    policy = RightDirectBCPolicy(
        obs_dim=int(ckpt["obs_dim"]),
        action_dim=int(ckpt["action_dim"]),
    ).to(device)

    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()

    obs_mean = np.asarray(ckpt["obs_mean"], dtype=np.float32)
    obs_std = np.asarray(ckpt["obs_std"], dtype=np.float32)
    action_mean = np.asarray(ckpt["action_mean"], dtype=np.float32)
    action_std = np.asarray(ckpt["action_std"], dtype=np.float32)

    action_min = np.asarray(ckpt["action_min"], dtype=np.float32)
    action_max = np.asarray(ckpt["action_max"], dtype=np.float32)

    if "action_actuator_names" in ckpt:
        action_actuator_names = [str(x) for x in ckpt["action_actuator_names"]]
    else:
        action_actuator_names = DEFAULT_ACTION_ACTUATORS

    print("=" * 80)
    print("Loaded right DIRECT BC policy")
    print("=" * 80)
    print("MODEL_PATH:", MODEL_PATH)
    print("best_epoch:", ckpt["best_epoch"])
    print("best_val_loss:", ckpt["best_val_loss"])
    print("val_mae:", ckpt["val_mae"])
    print("val_rmse:", ckpt["val_rmse"])
    print("action_actuator_names:", action_actuator_names)
    print("action_min:", action_min)
    print("action_max:", action_max)
    print("=" * 80)

    return (
        policy,
        obs_mean,
        obs_std,
        action_mean,
        action_std,
        action_min,
        action_max,
        action_actuator_names,
    )


def policy_action(policy, obs, obs_mean, obs_std, action_mean, action_std, action_min, action_max, device):
    obs_norm = ((obs - obs_mean) / obs_std).astype(np.float32)

    with torch.no_grad():
        x = torch.from_numpy(obs_norm).float().unsqueeze(0).to(device)
        pred_norm = policy(x).squeeze(0).detach().cpu().numpy()

    action = pred_norm * action_std + action_mean
    action = action.astype(np.float32)

    # 先限制到训练数据范围，防止控制发散
    action = np.clip(action, action_min, action_max)

    return action


def apply_direct_action(model, data, action, action_actuator_names):
    for value, name in zip(action, action_actuator_names):
        aid = rr.actuator_id(model, name)
        low, high = model.actuator_ctrlrange[aid]
        data.ctrl[aid] = np.clip(float(value), low, high)


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

    target = phase_target(model, data, site_name, phase_id)
    dist = float(np.linalg.norm(site_pos - target))

    cube_lift = float(cube_pos[2] - cube_initial_z)

    if phase_id == PHASE_PREGRASP:
        if phase_tick >= 18 and dist < 0.045:
            return True
        if phase_tick >= 85:
            return True
        return False

    if phase_id == PHASE_GRASP:
        if phase_tick >= 16 and dist < 0.035:
            return True
        if phase_tick >= 75:
            return True
        return False

    if phase_id == PHASE_CLOSE:
        if phase_tick >= 70:
            return True
        return False

    if phase_id == PHASE_LIFT:
        if phase_tick >= 24 and cube_lift > 0.055:
            return True
        if phase_tick >= 80:
            return True
        return False

    if phase_id == PHASE_PREPLACE:
        if phase_tick >= 18 and dist < 0.060:
            return True
        if phase_tick >= 90:
            return True
        return False

    if phase_id == PHASE_PLACE:
        if phase_tick >= 16 and dist < 0.050:
            return True
        if phase_tick >= 75:
            return True
        return False

    if phase_id == PHASE_RELEASE:
        if phase_tick >= 70:
            return True
        return False

    return False


def run_episode(
    policy,
    obs_mean,
    obs_std,
    action_mean,
    action_std,
    action_min,
    action_max,
    action_actuator_names,
    device,
    model,
    site_name,
    seed,
):
    data = mujoco.MjData(model)

    cube_pos = sample_cube_pos(seed)
    reset_scene_quiet(model, data, cube_pos)

    cube_initial = rr.get_body_pos(model, data, "orange_cube")
    cube_initial_z = float(cube_initial[2])

    phase_id = PHASE_PREGRASP
    phase_tick = 0
    total_tick = 0

    max_cube_z = cube_initial_z
    last_action = np.zeros(ACTION_DIM, dtype=np.float32)

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
            action_min=action_min,
            action_max=action_max,
            device=device,
        )

        last_action = action.copy()

        apply_direct_action(
            model=model,
            data=data,
            action=action,
            action_actuator_names=action_actuator_names,
        )

        tick_max_z = sim_control_steps(
            model=model,
            data=data,
            steps=CONTROL_STEPS,
        )

        max_cube_z = max(max_cube_z, tick_max_z)

        if VERBOSE_STEP:
            site_pos = rr.get_site_pos(model, data, site_name)
            target = phase_target(model, data, site_name, phase_id)
            dist = float(np.linalg.norm(site_pos - target))
            cube_now = rr.get_body_pos(model, data, "orange_cube")
            print(
                f"tick={total_tick:04d} "
                f"phase={PHASE_NAMES[phase_id]} "
                f"phase_tick={phase_tick:03d} "
                f"dist={dist:.4f} "
                f"cube_z_delta={cube_now[2] - cube_initial_z:.4f} "
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

        if total_tick >= 540:
            break

    final_cube = rr.get_body_pos(model, data, "orange_cube")
    frame_pos = rr.get_body_pos(model, data, "black_frame")

    lift_delta = float(max_cube_z - cube_initial_z)
    final_lift_delta = float(final_cube[2] - cube_initial_z)

    pick_success = bool(lift_delta > rr.LIFT_SUCCESS_DELTA_Z)

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
    print("Test right DIRECT BC policy v1")
    print("=" * 80)
    print("ROOT:", ROOT)
    print("MODEL_PATH:", MODEL_PATH)
    print("LOG_PATH:", LOG_PATH)
    print("NUM_EPISODES:", NUM_EPISODES)
    print("CUBE_X_RANGE:", CUBE_X_RANGE)
    print("CUBE_Y_RANGE:", CUBE_Y_RANGE)
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device:", device)
    if torch.cuda.is_available():
        print("cuda:", torch.cuda.get_device_name(0))

    (
        policy,
        obs_mean,
        obs_std,
        action_mean,
        action_std,
        action_min,
        action_max,
        action_actuator_names,
    ) = load_policy(device)

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
            action_min=action_min,
            action_max=action_max,
            action_actuator_names=action_actuator_names,
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
    print("Right DIRECT BC policy test 总结")
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
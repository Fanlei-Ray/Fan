from pathlib import Path
import sys
import time
import csv

import mujoco
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import right_rule_pick_place as rr


# ============================================================
# Build right-arm direct-actuator BC dataset v1
#
# 输入:
#   outputs/right_bc_v1/right_success_trials_v1.npz
#
# 输出:
#   outputs/right_bc_v1/right_bc_direct_dataset_v1.npz
#
# obs:
#   24 维，和之前一致
#
# action:
#   9 维:
#     right_joint1_ctrl
#     right_joint2_ctrl
#     right_joint3_ctrl
#     right_joint4_ctrl
#     right_joint5_ctrl
#     right_joint6_ctrl
#     right_joint7_ctrl
#     right_finger1_ctrl
#     lifter_ctrl
#
# 关键变化:
#   不再记录 TCP delta。
#   直接记录 expert 实际写进 data.ctrl 的 actuator 控制值。
#   之后测试时 policy 输出可以直接写 data.ctrl。
# ============================================================


RESULT_DIR = ROOT / "outputs" / "right_bc_v1"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SUCCESS_TRIALS_PATH = RESULT_DIR / "right_success_trials_v1.npz"
OUTPUT_DATASET_PATH = RESULT_DIR / "right_bc_direct_dataset_v1.npz"
OUTPUT_LOG_PATH = RESULT_DIR / "right_bc_direct_dataset_v1_log.csv"


RECORD_EVERY_N_STEPS = 40

OBS_DIM = 24
ACTION_DIM = 9


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


DIRECT_ACTION_ACTUATORS = [
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


def actuator_ctrl(model, data, name):
    aid = rr.actuator_id(model, name)
    return float(data.ctrl[aid])


def set_ctrl_clipped(model, data, name, value):
    aid = rr.actuator_id(model, name)
    low, high = model.actuator_ctrlrange[aid]
    data.ctrl[aid] = np.clip(float(value), low, high)


def get_direct_action(model, data):
    values = []

    for name in DIRECT_ACTION_ACTUATORS:
        values.append(actuator_ctrl(model, data, name))

    action = np.array(values, dtype=np.float32)
    assert action.shape == (ACTION_DIM,), action.shape
    return action


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

    assert obs.shape == (OBS_DIM,), obs.shape
    return obs


def append_sample(obs_list, action_list, model, data, site_name, phase_id):
    obs = make_obs(model, data, site_name, phase_id)
    action = get_direct_action(model, data)

    obs_list.append(obs)
    action_list.append(action)


def load_home_quiet(model, data):
    key_id = rr.maybe_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)

    rr.sync_position_actuators_to_qpos(model, data)
    mujoco.mj_forward(model, data)


def reset_scene_quiet(model, data, cube_pos):
    load_home_quiet(model, data)

    rr.set_free_body_pos(model, data, "orange_cube", cube_pos)
    set_ctrl_clipped(model, data, "right_finger1_ctrl", rr.RIGHT_FINGER_PRE_OPEN)
    set_ctrl_clipped(model, data, "lifter_ctrl", 0.0)

    mujoco.mj_forward(model, data)

    rr.sim_steps(model, data, steps=700, viewer=None, realtime=False)

    rr.set_free_body_pos(model, data, "orange_cube", cube_pos)
    set_ctrl_clipped(model, data, "right_finger1_ctrl", rr.RIGHT_FINGER_PRE_OPEN)
    set_ctrl_clipped(model, data, "lifter_ctrl", 0.0)

    mujoco.mj_forward(model, data)

    rr.sim_steps(model, data, steps=180, viewer=None, realtime=False)


def sim_steps_record(model, data, obs_list, action_list, site_name, phase_id, steps):
    for i in range(steps):
        if i % RECORD_EVERY_N_STEPS == 0:
            append_sample(
                obs_list=obs_list,
                action_list=action_list,
                model=model,
                data=data,
                site_name=site_name,
                phase_id=phase_id,
            )

        mujoco.mj_step(model, data)


def move_to_record(
    model,
    data,
    obs_list,
    action_list,
    site_name,
    phase_id,
    ctrl_targets,
    duration,
):
    start_ctrl = data.ctrl.copy()
    goal_ctrl = data.ctrl.copy()

    for name, value in ctrl_targets.items():
        aid = rr.actuator_id(model, name)
        low, high = model.actuator_ctrlrange[aid]
        goal_ctrl[aid] = np.clip(float(value), low, high)

    dt = model.opt.timestep
    steps = max(1, int(duration / dt))

    for i in range(steps):
        alpha = (i + 1) / steps
        alpha = 3.0 * alpha ** 2 - 2.0 * alpha ** 3

        data.ctrl[:] = (1.0 - alpha) * start_ctrl + alpha * goal_ctrl

        if i % RECORD_EVERY_N_STEPS == 0:
            append_sample(
                obs_list=obs_list,
                action_list=action_list,
                model=model,
                data=data,
                site_name=site_name,
                phase_id=phase_id,
            )

        mujoco.mj_step(model, data)


def ik_move_right_to_record(
    model,
    data,
    obs_list,
    action_list,
    site_name,
    phase_id,
    target_pos,
    joint_biases,
    duration,
):
    success, final_error, best_error, ctrl_targets = rr.solve_right_arm_ik(
        model=model,
        source_data=data,
        target_pos=target_pos,
        site_name=site_name,
    )

    ctrl_targets = rr.apply_joint_biases(
        model=model,
        ctrl_targets=ctrl_targets,
        joint_biases=joint_biases,
    )

    move_to_record(
        model=model,
        data=data,
        obs_list=obs_list,
        action_list=action_list,
        site_name=site_name,
        phase_id=phase_id,
        ctrl_targets=ctrl_targets,
        duration=duration,
    )

    sim_steps_record(
        model=model,
        data=data,
        obs_list=obs_list,
        action_list=action_list,
        site_name=site_name,
        phase_id=phase_id,
        steps=160,
    )

    actual_pos = rr.get_site_pos(model, data, site_name)
    actual_error = float(np.linalg.norm(np.asarray(target_pos, dtype=float) - actual_pos))

    return success, final_error, best_error, actual_error


def close_right_gripper_record(model, data, obs_list, action_list, site_name, duration):
    aid = rr.actuator_id(model, "right_finger1_ctrl")
    low, high = model.actuator_ctrlrange[aid]

    start_value = float(np.clip(rr.RIGHT_FINGER_PRE_OPEN, low, high))
    end_value = float(np.clip(rr.RIGHT_FINGER_CLOSE, low, high))

    data.ctrl[aid] = start_value

    sim_steps_record(
        model=model,
        data=data,
        obs_list=obs_list,
        action_list=action_list,
        site_name=site_name,
        phase_id=PHASE_CLOSE,
        steps=100,
    )

    dt = model.opt.timestep
    steps = max(1, int(duration / dt))

    for i in range(steps):
        alpha = (i + 1) / steps
        alpha = 3.0 * alpha ** 2 - 2.0 * alpha ** 3

        data.ctrl[aid] = (1.0 - alpha) * start_value + alpha * end_value

        if i % RECORD_EVERY_N_STEPS == 0:
            append_sample(
                obs_list=obs_list,
                action_list=action_list,
                model=model,
                data=data,
                site_name=site_name,
                phase_id=PHASE_CLOSE,
            )

        mujoco.mj_step(model, data)


def replay_one_success_to_direct_dataset(model, site_name, seed, cube_pos):
    data = mujoco.MjData(model)

    obs_list = []
    action_list = []

    reset_scene_quiet(model, data, cube_pos)

    cube_initial = rr.get_body_pos(model, data, "orange_cube")
    cube_initial_z = float(cube_initial[2])

    # 1. 右夹爪预张开
    move_to_record(
        model=model,
        data=data,
        obs_list=obs_list,
        action_list=action_list,
        site_name=site_name,
        phase_id=PHASE_PREGRASP,
        ctrl_targets={"right_finger1_ctrl": rr.RIGHT_FINGER_PRE_OPEN},
        duration=0.8,
    )

    # 2. pregrasp
    cube_now = rr.get_body_pos(model, data, "orange_cube")
    pregrasp_target = cube_now + BEST_CONFIG["pregrasp_offset"]

    ik_move_right_to_record(
        model=model,
        data=data,
        obs_list=obs_list,
        action_list=action_list,
        site_name=site_name,
        phase_id=PHASE_PREGRASP,
        target_pos=pregrasp_target,
        joint_biases=BEST_CONFIG["joint_biases"],
        duration=1.7,
    )

    # 3. grasp
    cube_now = rr.get_body_pos(model, data, "orange_cube")
    grasp_target = cube_now + BEST_CONFIG["grasp_offset"]

    ik_move_right_to_record(
        model=model,
        data=data,
        obs_list=obs_list,
        action_list=action_list,
        site_name=site_name,
        phase_id=PHASE_GRASP,
        target_pos=grasp_target,
        joint_biases=BEST_CONFIG["joint_biases"],
        duration=1.3,
    )

    # 4. close
    close_right_gripper_record(
        model=model,
        data=data,
        obs_list=obs_list,
        action_list=action_list,
        site_name=site_name,
        duration=2.2,
    )

    sim_steps_record(
        model=model,
        data=data,
        obs_list=obs_list,
        action_list=action_list,
        site_name=site_name,
        phase_id=PHASE_CLOSE,
        steps=300,
    )

    # 5. lift
    move_to_record(
        model=model,
        data=data,
        obs_list=obs_list,
        action_list=action_list,
        site_name=site_name,
        phase_id=PHASE_LIFT,
        ctrl_targets={
            "lifter_ctrl": rr.LIFTER_UP,
            "right_finger1_ctrl": rr.RIGHT_FINGER_CLOSE,
        },
        duration=1.5,
    )

    max_cube_z = cube_initial_z

    for i in range(500):
        if i % RECORD_EVERY_N_STEPS == 0:
            append_sample(
                obs_list=obs_list,
                action_list=action_list,
                model=model,
                data=data,
                site_name=site_name,
                phase_id=PHASE_LIFT,
            )

        mujoco.mj_step(model, data)

        cube_now = rr.get_body_pos(model, data, "orange_cube")
        max_cube_z = max(max_cube_z, float(cube_now[2]))

    cube_after_lift = rr.get_body_pos(model, data, "orange_cube")

    lift_delta = float(max_cube_z - cube_initial_z)
    final_lift_delta = float(cube_after_lift[2] - cube_initial_z)
    pick_success = lift_delta > rr.LIFT_SUCCESS_DELTA_Z

    if not pick_success:
        return None

    # 6. preplace
    frame_pos = rr.get_body_pos(model, data, "black_frame")
    preplace_target = frame_pos + BEST_CONFIG["preplace_offset"]

    ik_move_right_to_record(
        model=model,
        data=data,
        obs_list=obs_list,
        action_list=action_list,
        site_name=site_name,
        phase_id=PHASE_PREPLACE,
        target_pos=preplace_target,
        joint_biases={},
        duration=1.8,
    )

    # 7. place
    frame_pos = rr.get_body_pos(model, data, "black_frame")
    place_target = frame_pos + BEST_CONFIG["place_offset"]

    ik_move_right_to_record(
        model=model,
        data=data,
        obs_list=obs_list,
        action_list=action_list,
        site_name=site_name,
        phase_id=PHASE_PLACE,
        target_pos=place_target,
        joint_biases={},
        duration=1.4,
    )

    # 8. release
    move_to_record(
        model=model,
        data=data,
        obs_list=obs_list,
        action_list=action_list,
        site_name=site_name,
        phase_id=PHASE_RELEASE,
        ctrl_targets={"right_finger1_ctrl": rr.RIGHT_FINGER_OPEN},
        duration=1.0,
    )

    sim_steps_record(
        model=model,
        data=data,
        obs_list=obs_list,
        action_list=action_list,
        site_name=site_name,
        phase_id=PHASE_RELEASE,
        steps=500,
    )

    final_cube = rr.get_body_pos(model, data, "orange_cube")
    frame_pos = rr.get_body_pos(model, data, "black_frame")

    xy_dist = float(np.linalg.norm(final_cube[:2] - frame_pos[:2]))
    z_margin = float(final_cube[2] - frame_pos[2])

    place_success = bool((xy_dist < 0.055) and (z_margin > 0.005))

    if not place_success:
        return None

    obs = np.array(obs_list, dtype=np.float32)
    actions = np.array(action_list, dtype=np.float32)

    assert obs.ndim == 2 and obs.shape[1] == OBS_DIM, obs.shape
    assert actions.ndim == 2 and actions.shape[1] == ACTION_DIM, actions.shape
    assert obs.shape[0] == actions.shape[0]

    return {
        "seed": int(seed),
        "cube_pos": np.array(cube_pos, dtype=np.float32),
        "obs": obs,
        "actions": actions,
        "episode_length": int(len(obs)),
        "lift_delta": float(lift_delta),
        "final_lift_delta": float(final_lift_delta),
        "xy_dist": float(xy_dist),
        "z_margin": float(z_margin),
    }


def load_existing_dataset():
    if not OUTPUT_DATASET_PATH.exists():
        return [], [], [], [], [], [], [], [], []

    data = np.load(OUTPUT_DATASET_PATH, allow_pickle=True)

    obs_parts = [data["obs"]]
    action_parts = [data["actions"]]

    episode_lengths = list(data["episode_lengths"].astype(np.int32))
    seeds = list(data["seeds"].astype(np.int64))
    cube_positions = [x for x in data["cube_positions"]]

    lift_deltas = list(data["lift_deltas"].astype(np.float32))
    final_lift_deltas = list(data["final_lift_deltas"].astype(np.float32))
    xy_dists = list(data["xy_dists"].astype(np.float32))
    z_margins = list(data["z_margins"].astype(np.float32))

    return (
        obs_parts,
        action_parts,
        episode_lengths,
        seeds,
        cube_positions,
        lift_deltas,
        final_lift_deltas,
        xy_dists,
        z_margins,
    )


def save_dataset(
    obs_parts,
    action_parts,
    episode_lengths,
    seeds,
    cube_positions,
    lift_deltas,
    final_lift_deltas,
    xy_dists,
    z_margins,
):
    if not obs_parts:
        return

    obs = np.concatenate(obs_parts, axis=0).astype(np.float32)
    actions = np.concatenate(action_parts, axis=0).astype(np.float32)

    tmp_path = OUTPUT_DATASET_PATH.with_suffix(".tmp.npz")

    np.savez_compressed(
        tmp_path,
        obs=obs,
        actions=actions,
        episode_lengths=np.array(episode_lengths, dtype=np.int32),
        seeds=np.array(seeds, dtype=np.int64),
        cube_positions=np.array(cube_positions, dtype=np.float32),
        lift_deltas=np.array(lift_deltas, dtype=np.float32),
        final_lift_deltas=np.array(final_lift_deltas, dtype=np.float32),
        xy_dists=np.array(xy_dists, dtype=np.float32),
        z_margins=np.array(z_margins, dtype=np.float32),
        action_actuator_names=np.array(DIRECT_ACTION_ACTUATORS),
        record_every_n_steps=np.array(RECORD_EVERY_N_STEPS, dtype=np.int32),
        obs_dim=np.array(OBS_DIM, dtype=np.int32),
        action_dim=np.array(ACTION_DIM, dtype=np.int32),
    )

    tmp_path.replace(OUTPUT_DATASET_PATH)


def init_log():
    if OUTPUT_LOG_PATH.exists():
        return

    with open(OUTPUT_LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "index",
                "seed",
                "cube_x",
                "cube_y",
                "success",
                "episode_length",
                "lift_delta",
                "final_lift_delta",
                "xy_dist",
                "z_margin",
                "total_episodes",
                "total_samples",
            ]
        )


def append_log(index, seed, cube_pos, item, total_episodes, total_samples):
    with open(OUTPUT_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if item is None:
            writer.writerow(
                [
                    index,
                    seed,
                    float(cube_pos[0]),
                    float(cube_pos[1]),
                    False,
                    0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    total_episodes,
                    total_samples,
                ]
            )
        else:
            writer.writerow(
                [
                    index,
                    seed,
                    float(cube_pos[0]),
                    float(cube_pos[1]),
                    True,
                    int(item["episode_length"]),
                    float(item["lift_delta"]),
                    float(item["final_lift_delta"]),
                    float(item["xy_dist"]),
                    float(item["z_margin"]),
                    total_episodes,
                    total_samples,
                ]
            )


def count_samples(obs_parts):
    return int(sum(len(x) for x in obs_parts))


def main():
    print("=" * 80)
    print("Build right BC DIRECT actuator dataset v1")
    print("=" * 80)
    print("SUCCESS_TRIALS_PATH:", SUCCESS_TRIALS_PATH)
    print("OUTPUT_DATASET_PATH:", OUTPUT_DATASET_PATH)
    print("OUTPUT_LOG_PATH:", OUTPUT_LOG_PATH)
    print("RECORD_EVERY_N_STEPS:", RECORD_EVERY_N_STEPS)
    print("OBS_DIM:", OBS_DIM)
    print("ACTION_DIM:", ACTION_DIM)
    print("DIRECT_ACTION_ACTUATORS:", DIRECT_ACTION_ACTUATORS)
    print("=" * 80)

    if not SUCCESS_TRIALS_PATH.exists():
        raise FileNotFoundError(f"找不到成功 seed 文件：{SUCCESS_TRIALS_PATH}")

    success_data = np.load(SUCCESS_TRIALS_PATH, allow_pickle=True)

    trial_seeds = success_data["seeds"].astype(np.int64)
    trial_cube_positions = success_data["cube_positions"].astype(np.float32)

    print("成功 seed 数量:", len(trial_seeds))

    (
        obs_parts,
        action_parts,
        episode_lengths,
        seeds,
        cube_positions,
        lift_deltas,
        final_lift_deltas,
        xy_dists,
        z_margins,
    ) = load_existing_dataset()

    done_seeds = set(int(s) for s in seeds)

    print("已有 direct episodes:", len(seeds))
    print("已有 direct samples:", count_samples(obs_parts))

    init_log()

    model = mujoco.MjModel.from_xml_path(str(rr.XML_PATH))
    site_name = rr.choose_right_site(model)

    print("使用右臂 site:", site_name)

    start_time = time.time()

    for i, (seed, cube_pos) in enumerate(zip(trial_seeds, trial_cube_positions), start=1):
        seed = int(seed)

        if seed in done_seeds:
            continue

        item = replay_one_success_to_direct_dataset(
            model=model,
            site_name=site_name,
            seed=seed,
            cube_pos=cube_pos,
        )

        if item is not None:
            obs_parts.append(item["obs"])
            action_parts.append(item["actions"])

            episode_lengths.append(item["episode_length"])
            seeds.append(item["seed"])
            cube_positions.append(item["cube_pos"])
            lift_deltas.append(item["lift_delta"])
            final_lift_deltas.append(item["final_lift_delta"])
            xy_dists.append(item["xy_dist"])
            z_margins.append(item["z_margin"])

            done_seeds.add(seed)

            save_dataset(
                obs_parts=obs_parts,
                action_parts=action_parts,
                episode_lengths=episode_lengths,
                seeds=seeds,
                cube_positions=cube_positions,
                lift_deltas=lift_deltas,
                final_lift_deltas=final_lift_deltas,
                xy_dists=xy_dists,
                z_margins=z_margins,
            )

        total_samples = count_samples(obs_parts)
        total_episodes = len(seeds)

        append_log(
            index=i,
            seed=seed,
            cube_pos=cube_pos,
            item=item,
            total_episodes=total_episodes,
            total_samples=total_samples,
        )

        mark = "O" if item is not None else "X"
        ep_len = 0 if item is None else item["episode_length"]
        lift = 0.0 if item is None else item["lift_delta"]
        final_lift = 0.0 if item is None else item["final_lift_delta"]
        xy = 0.0 if item is None else item["xy_dist"]
        z_margin = 0.0 if item is None else item["z_margin"]

        print(
            f"{mark} {i:03d}/{len(trial_seeds):03d} "
            f"seed={seed} "
            f"cube=({cube_pos[0]:.3f},{cube_pos[1]:.3f}) "
            f"ep_len={ep_len} "
            f"lift={lift:.4f} "
            f"final_lift={final_lift:.4f} "
            f"xy={xy:.4f} "
            f"z_margin={z_margin:.4f} "
            f"episodes={total_episodes} "
            f"samples={total_samples}"
        )

    save_dataset(
        obs_parts=obs_parts,
        action_parts=action_parts,
        episode_lengths=episode_lengths,
        seeds=seeds,
        cube_positions=cube_positions,
        lift_deltas=lift_deltas,
        final_lift_deltas=final_lift_deltas,
        xy_dists=xy_dists,
        z_margins=z_margins,
    )

    elapsed = time.time() - start_time

    final_data = np.load(OUTPUT_DATASET_PATH, allow_pickle=True)

    print("")
    print("=" * 80)
    print("Right BC DIRECT dataset build 总结")
    print("=" * 80)
    print("episodes:", len(final_data["episode_lengths"]))
    print("obs shape:", final_data["obs"].shape)
    print("actions shape:", final_data["actions"].shape)
    print("episode_lengths shape:", final_data["episode_lengths"].shape)
    print("action_actuator_names:", final_data["action_actuator_names"])
    print("action min:", final_data["actions"].min(axis=0))
    print("action max:", final_data["actions"].max(axis=0))
    print("action mean:", final_data["actions"].mean(axis=0))
    print("耗时:", f"{elapsed / 60:.1f} min")
    print("保存文件:", OUTPUT_DATASET_PATH)
    print("日志文件:", OUTPUT_LOG_PATH)


if __name__ == "__main__":
    main()
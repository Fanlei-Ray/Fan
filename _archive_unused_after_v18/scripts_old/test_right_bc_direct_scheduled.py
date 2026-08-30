from pathlib import Path
import sys
import csv
import time

import mujoco
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import right_rule_pick_place as rr
import test_right_bc_direct as base


# ============================================================
# Test right DIRECT BC v1, but schedule finger/lifter by phase
#
# 目的：
#   验证 0/30 是不是因为 policy 没有正确闭合夹爪/抬 lifter。
#
# 控制方式：
#   action[0:7] = policy 输出右臂 7 关节 actuator ctrl
#   action[7]   = 规则时序控制 right_finger1_ctrl
#   action[8]   = 规则时序控制 lifter_ctrl
# ============================================================


RESULT_DIR = ROOT / "outputs" / "right_bc_v1"
LOG_PATH = RESULT_DIR / "right_bc_direct_test_v1_scheduled_log.csv"

NUM_EPISODES = 30
SEED_START = 61001

CONTROL_STEPS = base.CONTROL_STEPS


def scheduled_finger_lifter(phase_id):
    if phase_id in [base.PHASE_PREGRASP, base.PHASE_GRASP]:
        return rr.RIGHT_FINGER_PRE_OPEN, 0.0

    if phase_id == base.PHASE_CLOSE:
        return rr.RIGHT_FINGER_CLOSE, 0.0

    if phase_id in [base.PHASE_LIFT, base.PHASE_PREPLACE, base.PHASE_PLACE]:
        return rr.RIGHT_FINGER_CLOSE, rr.LIFTER_UP

    if phase_id == base.PHASE_RELEASE:
        return rr.RIGHT_FINGER_OPEN, rr.LIFTER_UP

    return rr.RIGHT_FINGER_PRE_OPEN, 0.0


def apply_scheduled_action(model, data, action, action_actuator_names, phase_id):
    action = action.copy()

    finger_value, lifter_value = scheduled_finger_lifter(phase_id)

    for i, name in enumerate(action_actuator_names):
        if name == "right_finger1_ctrl":
            action[i] = finger_value
        elif name == "lifter_ctrl":
            action[i] = lifter_value

    base.apply_direct_action(
        model=model,
        data=data,
        action=action,
        action_actuator_names=action_actuator_names,
    )

    return action


def run_episode_scheduled(
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

    cube_pos = base.sample_cube_pos(seed)
    base.reset_scene_quiet(model, data, cube_pos)

    cube_initial = rr.get_body_pos(model, data, "orange_cube")
    cube_initial_z = float(cube_initial[2])

    phase_id = base.PHASE_PREGRASP
    phase_tick = 0
    total_tick = 0

    max_cube_z = cube_initial_z
    last_action = np.zeros(base.ACTION_DIM, dtype=np.float32)

    while True:
        total_tick += 1
        phase_tick += 1

        obs = base.make_obs(model, data, site_name, phase_id)

        action = base.policy_action(
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

        action = apply_scheduled_action(
            model=model,
            data=data,
            action=action,
            action_actuator_names=action_actuator_names,
            phase_id=phase_id,
        )

        last_action = action.copy()

        tick_max_z = base.sim_control_steps(
            model=model,
            data=data,
            steps=CONTROL_STEPS,
        )

        max_cube_z = max(max_cube_z, tick_max_z)

        advance = base.should_advance_phase(
            model=model,
            data=data,
            site_name=site_name,
            phase_id=phase_id,
            phase_tick=phase_tick,
            cube_initial_z=cube_initial_z,
        )

        if advance:
            if phase_id == base.PHASE_RELEASE:
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
        "last_phase": base.PHASE_NAMES.get(phase_id, str(phase_id)),
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
    print("Test right DIRECT BC v1 with scheduled finger/lifter")
    print("=" * 80)
    print("MODEL_PATH:", base.MODEL_PATH)
    print("LOG_PATH:", LOG_PATH)
    print("NUM_EPISODES:", NUM_EPISODES)
    print("SEED_START:", SEED_START)
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
    ) = base.load_policy(device)

    model = mujoco.MjModel.from_xml_path(str(rr.XML_PATH))
    site_name = rr.choose_right_site(model)

    print("使用右臂 site:", site_name)
    print("action_actuator_names:", action_actuator_names)

    init_log()

    results = []
    start_time = time.time()

    for ep in range(1, NUM_EPISODES + 1):
        seed = SEED_START + ep

        result = run_episode_scheduled(
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
    print("Right DIRECT BC scheduled test 总结")
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
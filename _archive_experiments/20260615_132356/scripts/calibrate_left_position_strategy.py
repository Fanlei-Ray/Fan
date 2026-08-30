import numpy as np

from openarm_pickplace_env import OpenArmPickPlaceEnv
import right_pick_place as pp


TEST_POINTS = [
    # 左臂失败带
    (0.497, 0.050),
    (0.516, 0.050),
    (0.534, 0.050),
    (0.497, 0.075),
    (0.516, 0.075),
    (0.534, 0.075),

    # 对照点
    (0.516, 0.000),
    (0.516, -0.050),
]


STRATEGIES = [
    {
        "name": "normal_v4_like",
        "pregrasp_offset": np.array([-0.005, 0.000, 0.100], dtype=np.float32),
        "grasp_offset": np.array([-0.010, 0.000, -0.005], dtype=np.float32),
        "finger_open": 0.445,
        "finger_close": 0.0,
        "pre_thresh": 0.030,
        "grasp_thresh": 0.025,
        "open_steps": 18,
        "pre_steps": 85,
        "grasp_steps": 95,
        "close_steps": 45,
        "hold_steps": 12,
        "lift_steps": 60,
    },
    {
        "name": "higher_grasp",
        "pregrasp_offset": np.array([-0.005, 0.000, 0.115], dtype=np.float32),
        "grasp_offset": np.array([-0.010, 0.000, 0.010], dtype=np.float32),
        "finger_open": 0.445,
        "finger_close": 0.0,
        "pre_thresh": 0.035,
        "grasp_thresh": 0.030,
        "open_steps": 20,
        "pre_steps": 100,
        "grasp_steps": 110,
        "close_steps": 55,
        "hold_steps": 16,
        "lift_steps": 70,
    },
    {
        "name": "away_from_frame_y_minus",
        "pregrasp_offset": np.array([-0.005, -0.018, 0.110], dtype=np.float32),
        "grasp_offset": np.array([-0.010, -0.018, 0.000], dtype=np.float32),
        "finger_open": 0.445,
        "finger_close": 0.0,
        "pre_thresh": 0.035,
        "grasp_thresh": 0.030,
        "open_steps": 20,
        "pre_steps": 105,
        "grasp_steps": 115,
        "close_steps": 60,
        "hold_steps": 18,
        "lift_steps": 75,
    },
    {
        "name": "toward_frame_y_plus",
        "pregrasp_offset": np.array([-0.005, 0.018, 0.110], dtype=np.float32),
        "grasp_offset": np.array([-0.010, 0.018, 0.000], dtype=np.float32),
        "finger_open": 0.445,
        "finger_close": 0.0,
        "pre_thresh": 0.035,
        "grasp_thresh": 0.030,
        "open_steps": 20,
        "pre_steps": 105,
        "grasp_steps": 115,
        "close_steps": 60,
        "hold_steps": 18,
        "lift_steps": 75,
    },
    {
        "name": "x_plus",
        "pregrasp_offset": np.array([0.010, 0.000, 0.110], dtype=np.float32),
        "grasp_offset": np.array([0.010, 0.000, 0.000], dtype=np.float32),
        "finger_open": 0.445,
        "finger_close": 0.0,
        "pre_thresh": 0.035,
        "grasp_thresh": 0.030,
        "open_steps": 20,
        "pre_steps": 105,
        "grasp_steps": 115,
        "close_steps": 60,
        "hold_steps": 18,
        "lift_steps": 75,
    },
    {
        "name": "x_minus_more",
        "pregrasp_offset": np.array([-0.025, 0.000, 0.110], dtype=np.float32),
        "grasp_offset": np.array([-0.025, 0.000, 0.000], dtype=np.float32),
        "finger_open": 0.445,
        "finger_close": 0.0,
        "pre_thresh": 0.035,
        "grasp_thresh": 0.030,
        "open_steps": 20,
        "pre_steps": 105,
        "grasp_steps": 115,
        "close_steps": 60,
        "hold_steps": 18,
        "lift_steps": 75,
    },
    {
        "name": "full_open_slow_close",
        "pregrasp_offset": np.array([-0.005, 0.000, 0.115], dtype=np.float32),
        "grasp_offset": np.array([-0.010, 0.000, 0.005], dtype=np.float32),
        "finger_open": 0.490,
        "finger_close": 0.0,
        "pre_thresh": 0.035,
        "grasp_thresh": 0.030,
        "open_steps": 25,
        "pre_steps": 110,
        "grasp_steps": 125,
        "close_steps": 80,
        "hold_steps": 25,
        "lift_steps": 85,
    },
    {
        "name": "very_conservative",
        "pregrasp_offset": np.array([-0.005, -0.010, 0.130], dtype=np.float32),
        "grasp_offset": np.array([-0.010, -0.010, 0.010], dtype=np.float32),
        "finger_open": 0.490,
        "finger_close": 0.0,
        "pre_thresh": 0.040,
        "grasp_thresh": 0.035,
        "open_steps": 30,
        "pre_steps": 130,
        "grasp_steps": 140,
        "close_steps": 90,
        "hold_steps": 30,
        "lift_steps": 90,
    },
]


FRAME_SKIP = 30
MAX_TCP_DELTA = 0.024
MAX_FINGER_DELTA = 0.040
MAX_LIFTER_DELTA = 0.025

MAX_EPISODE_STEPS = 500


def make_env(x, y):
    env = OpenArmPickPlaceEnv(
        render_mode=None,
        randomize_cube=True,
        cube_x_range=(float(x), float(x)),
        cube_y_range=(float(y), float(y)),
        max_episode_steps=MAX_EPISODE_STEPS,
    )

    env.frame_skip = FRAME_SKIP
    env.max_tcp_delta = MAX_TCP_DELTA
    env.max_finger_delta = MAX_FINGER_DELTA
    env.max_lifter_delta = MAX_LIFTER_DELTA

    return env


def action_to_targets(env, obs, info, tcp_target=None, finger_target=None, lifter_target=None):
    action = np.zeros(5, dtype=np.float32)

    tcp_pos = info["tcp_pos"].astype(np.float32)
    finger_ctrl = float(obs[22])
    lifter_ctrl = float(obs[23])

    if tcp_target is not None:
        delta = np.asarray(tcp_target, dtype=np.float32) - tcp_pos
        action[:3] = np.clip(delta / env.max_tcp_delta, -1.0, 1.0)

    if finger_target is not None:
        action[3] = np.clip(
            (float(finger_target) - finger_ctrl) / env.max_finger_delta,
            -1.0,
            1.0,
        )

    if lifter_target is not None:
        action[4] = np.clip(
            (float(lifter_target) - lifter_ctrl) / env.max_lifter_delta,
            -1.0,
            1.0,
        )

    return action.astype(np.float32)


def run_until(env, obs, info, strategy, phase_name, tcp_target=None, finger_target=None, lifter_target=None, max_steps=80, dist_thresh=None):
    total_reward = 0.0
    final_reason = "max_steps"

    for step in range(max_steps):
        action = action_to_targets(
            env,
            obs,
            info,
            tcp_target=tcp_target,
            finger_target=finger_target,
            lifter_target=lifter_target,
        )

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        if dist_thresh is not None and tcp_target is not None:
            dist = float(np.linalg.norm(info["tcp_pos"] - tcp_target))
            if dist < dist_thresh:
                final_reason = f"{phase_name}_dist_ok"
                break

        if truncated:
            final_reason = "truncated"
            break

    return obs, info, total_reward, final_reason


def run_one_strategy(x, y, strategy, seed):
    env = make_env(x, y)

    obs, info = env.reset(seed=seed)

    total_reward = 0.0
    reason_log = []

    # 记录稳定后的 cube 位置，而不是直接用 reset 的位置
    for _ in range(5):
        action = np.zeros(5, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

    cube_pick_pos = info["cube_pos"].copy()
    cube_initial_z = float(cube_pick_pos[2])

    # open gripper
    obs, info, r, reason = run_until(
        env,
        obs,
        info,
        strategy,
        phase_name="open",
        tcp_target=info["tcp_pos"],
        finger_target=strategy["finger_open"],
        lifter_target=pp.LIFTER_HOME,
        max_steps=strategy["open_steps"],
        dist_thresh=None,
    )
    total_reward += r
    reason_log.append(reason)

    # move pregrasp
    pregrasp_target = cube_pick_pos + strategy["pregrasp_offset"]

    obs, info, r, reason = run_until(
        env,
        obs,
        info,
        strategy,
        phase_name="pregrasp",
        tcp_target=pregrasp_target,
        finger_target=strategy["finger_open"],
        lifter_target=pp.LIFTER_HOME,
        max_steps=strategy["pre_steps"],
        dist_thresh=strategy["pre_thresh"],
    )
    total_reward += r
    reason_log.append(reason)

    # move grasp
    grasp_target = cube_pick_pos + strategy["grasp_offset"]

    obs, info, r, reason = run_until(
        env,
        obs,
        info,
        strategy,
        phase_name="grasp",
        tcp_target=grasp_target,
        finger_target=strategy["finger_open"],
        lifter_target=pp.LIFTER_HOME,
        max_steps=strategy["grasp_steps"],
        dist_thresh=strategy["grasp_thresh"],
    )
    total_reward += r
    reason_log.append(reason)

    # close gripper
    obs, info, r, reason = run_until(
        env,
        obs,
        info,
        strategy,
        phase_name="close",
        tcp_target=grasp_target,
        finger_target=strategy["finger_close"],
        lifter_target=pp.LIFTER_HOME,
        max_steps=strategy["close_steps"],
        dist_thresh=None,
    )
    total_reward += r
    reason_log.append(reason)

    # hold
    obs, info, r, reason = run_until(
        env,
        obs,
        info,
        strategy,
        phase_name="hold",
        tcp_target=grasp_target,
        finger_target=strategy["finger_close"],
        lifter_target=pp.LIFTER_HOME,
        max_steps=strategy["hold_steps"],
        dist_thresh=None,
    )
    total_reward += r
    reason_log.append(reason)

    # lift
    obs, info, r, reason = run_until(
        env,
        obs,
        info,
        strategy,
        phase_name="lift",
        tcp_target=info["tcp_pos"],
        finger_target=strategy["finger_close"],
        lifter_target=pp.LIFTER_UP,
        max_steps=strategy["lift_steps"],
        dist_thresh=None,
    )
    total_reward += r
    reason_log.append(reason)

    cube_final = info["cube_pos"].copy()
    lift_delta = float(cube_final[2] - cube_initial_z)
    lifted = bool(info["cube_lifted"]) or lift_delta > 0.015

    env.close()

    result = {
        "success": lifted,
        "lifted": lifted,
        "lift_delta": lift_delta,
        "cube_initial": cube_pick_pos,
        "cube_final": cube_final,
        "tcp_final": info["tcp_pos"].copy(),
        "return": total_reward,
        "reason_log": reason_log,
    }

    return result


def main():
    print("=" * 80)
    print("左臂位置抓取策略校准")
    print("=" * 80)
    print("这次不用 pose IK，不转手腕，只测试原 env / 原位置 IK 下的抓取参数。")
    print("目标：找出 y=+0.050 / +0.075 失败带是否能靠高度/偏移/夹爪参数修复。")
    print("=" * 80)

    summary = []

    for x, y in TEST_POINTS:
        print("")
        print("=" * 80)
        print(f"测试点 x={x:.3f}, y={y:.3f}")
        print("=" * 80)

        best = None

        for i, strategy in enumerate(STRATEGIES):
            seed = 81000 + int(round(x * 1000)) * 10 + int(round((y + 1.0) * 1000)) + i

            result = run_one_strategy(
                x=x,
                y=y,
                strategy=strategy,
                seed=seed,
            )

            mark = "O" if result["success"] else "X"

            print(
                f"{mark} | "
                f"strategy={strategy['name']:24s} | "
                f"lift_delta={result['lift_delta']:+.4f} | "
                f"cube_final={np.array2string(result['cube_final'], precision=3)} | "
                f"tcp_final={np.array2string(result['tcp_final'], precision=3)}"
            )

            if result["success"] and best is None:
                best = {
                    "strategy": strategy,
                    "result": result,
                }

        if best is None:
            summary.append((x, y, False, None))
        else:
            summary.append((x, y, True, best))

    print("")
    print("=" * 80)
    print("左臂位置策略校准总结")
    print("=" * 80)

    for x, y, success, best in summary:
        if not success:
            print(f"x={x:.3f}, y={y:.3f}: X no successful pick/lift")
        else:
            s = best["strategy"]
            r = best["result"]

            print(
                f"x={x:.3f}, y={y:.3f}: O "
                f"strategy={s['name']}, "
                f"pre={np.array2string(s['pregrasp_offset'], precision=3)}, "
                f"grasp={np.array2string(s['grasp_offset'], precision=3)}, "
                f"finger_open={s['finger_open']}, "
                f"lift_delta={r['lift_delta']:+.4f}"
            )

    success_count = sum(1 for _, _, success, _ in summary if success)

    print("")
    print(f"成功点数: {success_count}/{len(summary)}")

    print("")
    print("判断：")
    print("1. 如果失败带出现 O，下一步把对应策略写进 expert，做 v6/v7 数据。")
    print("2. 如果对照点 O 但失败带 X，说明失败带需要避障/姿态，不是简单位置参数。")
    print("3. 如果连对照点都大面积 X，说明这个校准脚本和 v4 expert 有差异，需要回滚到 v4 逻辑。")


if __name__ == "__main__":
    main()
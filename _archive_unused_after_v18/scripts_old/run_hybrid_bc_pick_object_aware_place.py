from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from openarm_pickplace_env import OpenArmPickPlaceEnv
import right_pick_place as pp


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "bc_policy_phase_v4.pt"

NUM_EPISODES = 30
MAX_STEPS = 420
RENDER = False

CUBE_X_RANGE = (0.48, 0.57)
CUBE_Y_RANGE = (-0.06, 0.06)

LEFT_FINGER_OPEN = getattr(pp, "LEFT_FINGER_OPEN", 0.49)
LEFT_FINGER_PRE_OPEN = getattr(pp, "LEFT_FINGER_PRE_OPEN", 0.445)
LEFT_FINGER_CLOSE = getattr(pp, "LEFT_FINGER_CLOSE", 0.0)

LIFTER_UP = getattr(pp, "LIFTER_UP", 0.11)
LIFTER_HOME = getattr(pp, "LIFTER_HOME", 0.0)

PICK_PREGRASP_OFFSET = getattr(pp, "PICK_PREGRASP_OFFSET", np.array([-0.005, 0.0, 0.10], dtype=np.float32))
PICK_GRASP_OFFSET = getattr(pp, "PICK_GRASP_OFFSET", np.array([-0.010, 0.0, -0.005], dtype=np.float32))

# 注意：这是“方块目标高度”，不是 TCP 目标高度
CUBE_PREPLACE_OFFSET = np.array([0.0, 0.0, 0.13], dtype=np.float32)
CUBE_RELEASE_OFFSET = np.array([0.0, 0.0, 0.045], dtype=np.float32)


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


class PickController:
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
        finger_ctrl = float(obs[22])
        lifter_ctrl = float(obs[23])

        if self.pick_cube_pos is None:
            self.pick_cube_pos = info["cube_pos"].copy()

        if self.phase == "open_gripper":
            if finger_ctrl >= LEFT_FINGER_PRE_OPEN - 0.02 or self.phase_steps > 18:
                self.pick_cube_pos = info["cube_pos"].copy()
                self._switch("move_pregrasp")
            return

        if self.phase == "move_pregrasp":
            target = self.pick_cube_pos + PICK_PREGRASP_OFFSET
            if np.linalg.norm(tcp_pos - target) < 0.030 or self.phase_steps > 85:
                self._switch("move_grasp")
            return

        if self.phase == "move_grasp":
            target = self.pick_cube_pos + PICK_GRASP_OFFSET
            if np.linalg.norm(tcp_pos - target) < 0.025 or self.phase_steps > 95:
                self._switch("close_gripper")
            return

        if self.phase == "close_gripper":
            if finger_ctrl <= LEFT_FINGER_CLOSE + 0.04 or self.phase_steps > 40:
                self._switch("hold_grasp")
            return

        if self.phase == "hold_grasp":
            if self.phase_steps > 10:
                self._switch("lift_object")
            return

        if self.phase == "lift_object":
            lift_delta = float(info["cube_pos"][2] - self.pick_cube_pos[2])
            lifted_ok = bool(info["cube_lifted"]) or lift_delta > 0.025
            lifter_ok = lifter_ctrl >= LIFTER_UP - 0.015
            time_ok = self.phase_steps > 8

            if (lifted_ok and lifter_ok and time_ok) or self.phase_steps > 65:
                self._switch("object_aware_preplace")
            return

        if self.phase.startswith("object_aware"):
            return


class ObjectAwarePlaceController:
    def __init__(self):
        self.phase = "object_aware_preplace"
        self.phase_steps = 0

    def reset(self):
        self.phase = "object_aware_preplace"
        self.phase_steps = 0

    def _switch(self, next_phase):
        self.phase = next_phase
        self.phase_steps = 0

    def update(self, obs, info):
        self.phase_steps += 1

        cube_pos = info["cube_pos"]
        frame_pos = info["frame_pos"]
        finger_ctrl = float(obs[22])

        cube_preplace_target = frame_pos + CUBE_PREPLACE_OFFSET
        cube_release_target = frame_pos + CUBE_RELEASE_OFFSET

        if self.phase == "object_aware_preplace":
            dist = float(np.linalg.norm(cube_pos[:2] - cube_preplace_target[:2]))
            z_ok = cube_pos[2] > frame_pos[2] + 0.07

            if (dist < 0.045 and z_ok) or self.phase_steps > 120:
                self._switch("object_aware_release")
            return

        if self.phase == "object_aware_release":
            dist = float(np.linalg.norm(cube_pos[:2] - cube_release_target[:2]))
            z_close = abs(float(cube_pos[2] - cube_release_target[2])) < 0.05

            if (dist < 0.035 and z_close) or self.phase_steps > 110:
                self._switch("object_aware_open")
            return

        if self.phase == "object_aware_open":
            if finger_ctrl >= LEFT_FINGER_OPEN - 0.04 or self.phase_steps > 55:
                self._switch("object_aware_done")
            return

        if self.phase == "object_aware_done":
            return


def load_policy(path, device):
    if not path.exists():
        raise FileNotFoundError(f"找不到模型：{path}")

    checkpoint = safe_torch_load(path, device)

    model = PhaseBCPolicy(
        input_dim=int(checkpoint["input_dim"]),
        act_dim=int(checkpoint["act_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    phase_names = list(checkpoint["phase_names"])
    obs_aug_mean = checkpoint["obs_aug_mean"].to(device)
    obs_aug_std = checkpoint["obs_aug_std"].to(device)
    action_mean = checkpoint["action_mean"].to(device)
    action_std = checkpoint["action_std"].to(device)
    env_config = checkpoint.get("env_config", {})

    print("已加载 BC 模型:", path)
    print("best_epoch:", checkpoint.get("epoch"))
    print("best_val_loss:", checkpoint.get("best_val_loss"))
    print("val_mae:", checkpoint.get("val_mae"))
    print("env_config:", env_config)

    return model, phase_names, obs_aug_mean, obs_aug_std, action_mean, action_std, env_config


def bc_action(model, obs, phase, phase_names, obs_aug_mean, obs_aug_std, action_mean, action_std, device):
    p = phase_onehot(phase, phase_names)
    obs_aug = np.concatenate([obs.astype(np.float32), p], axis=0).astype(np.float32)

    obs_tensor = torch.from_numpy(obs_aug).float().to(device)
    obs_norm = (obs_tensor - obs_aug_mean) / obs_aug_std

    with torch.no_grad():
        action_norm = model(obs_norm.unsqueeze(0)).squeeze(0)
        action = action_norm * action_std + action_mean

    action = action.cpu().numpy().astype(np.float32)
    return np.clip(action, -1.0, 1.0)


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


def object_aware_tcp_target(info, desired_cube_pos):
    """
    核心修正：
    不让 TCP 直接去 frame 上方，而是让 cube 去 frame 上方。

    如果当前：
        tcp_pos - cube_pos = 抓取偏移

    那么要让 cube_pos -> desired_cube_pos，
    就应该让：
        tcp_target = desired_cube_pos + (tcp_pos - cube_pos)
    """
    tcp_pos = info["tcp_pos"].astype(np.float32)
    cube_pos = info["cube_pos"].astype(np.float32)

    tcp_minus_cube = tcp_pos - cube_pos
    return desired_cube_pos.astype(np.float32) + tcp_minus_cube


def object_aware_place_action(env, obs, info, controller):
    frame_pos = info["frame_pos"].astype(np.float32)

    if controller.phase == "object_aware_preplace":
        desired_cube_pos = frame_pos + CUBE_PREPLACE_OFFSET
        tcp_target = object_aware_tcp_target(info, desired_cube_pos)

        return action_to_targets(
            env,
            obs,
            info,
            tcp_target=tcp_target,
            finger_target=LEFT_FINGER_CLOSE,
            lifter_target=LIFTER_UP,
        )

    if controller.phase == "object_aware_release":
        desired_cube_pos = frame_pos + CUBE_RELEASE_OFFSET
        tcp_target = object_aware_tcp_target(info, desired_cube_pos)

        return action_to_targets(
            env,
            obs,
            info,
            tcp_target=tcp_target,
            finger_target=LEFT_FINGER_CLOSE,
            lifter_target=LIFTER_UP,
        )

    if controller.phase == "object_aware_open":
        desired_cube_pos = frame_pos + CUBE_RELEASE_OFFSET
        tcp_target = object_aware_tcp_target(info, desired_cube_pos)

        return action_to_targets(
            env,
            obs,
            info,
            tcp_target=tcp_target,
            finger_target=LEFT_FINGER_OPEN,
            lifter_target=LIFTER_UP,
        )

    if controller.phase == "object_aware_done":
        return action_to_targets(
            env,
            obs,
            info,
            tcp_target=info["tcp_pos"],
            finger_target=LEFT_FINGER_OPEN,
            lifter_target=LIFTER_UP,
        )

    raise RuntimeError(f"未知 object-aware phase: {controller.phase}")


def make_env(env_config):
    env = OpenArmPickPlaceEnv(
        render_mode="human" if RENDER else None,
        randomize_cube=True,
        cube_x_range=CUBE_X_RANGE,
        cube_y_range=CUBE_Y_RANGE,
        max_episode_steps=MAX_STEPS,
    )

    if "frame_skip" in env_config:
        env.frame_skip = int(env_config["frame_skip"])

    if "max_tcp_delta" in env_config:
        env.max_tcp_delta = float(env_config["max_tcp_delta"])

    if "max_finger_delta" in env_config:
        env.max_finger_delta = float(env_config["max_finger_delta"])

    if "max_lifter_delta" in env_config:
        env.max_lifter_delta = float(env_config["max_lifter_delta"])

    return env


def run_episode(env, ep, model, phase_names, obs_aug_mean, obs_aug_std, action_mean, action_std, device):
    obs, info = env.reset(seed=13000 + ep)

    pick = PickController()
    pick.reset(info)

    place = ObjectAwarePlaceController()
    place.reset()

    using_place = False

    total_reward = 0.0
    success_ever = False
    lifted_ever = False

    initial_cube_z = float(info["cube_pos"][2])
    max_cube_z = initial_cube_z

    final_info = info

    print("")
    print("=" * 60)
    print(f"Object-aware Hybrid Episode {ep}/{NUM_EPISODES}")
    print("初始 cube_pos:", np.array2string(info["cube_pos"], precision=4))
    print("初始 frame_pos:", np.array2string(info["frame_pos"], precision=4))

    for step in range(MAX_STEPS):
        if not using_place:
            phase = pick.phase

            if phase == "object_aware_preplace":
                using_place = True
                place.reset()
                phase = place.phase
                action = object_aware_place_action(env, obs, info, place)
            else:
                action = bc_action(
                    model=model,
                    obs=obs,
                    phase=phase,
                    phase_names=phase_names,
                    obs_aug_mean=obs_aug_mean,
                    obs_aug_std=obs_aug_std,
                    action_mean=action_mean,
                    action_std=action_std,
                    device=device,
                )
        else:
            phase = place.phase
            action = object_aware_place_action(env, obs, info, place)

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        final_info = info

        success_ever = success_ever or bool(info["is_success"])
        lifted_ever = lifted_ever or bool(info["cube_lifted"])
        max_cube_z = max(max_cube_z, float(info["cube_pos"][2]))

        if using_place:
            place.update(obs, info)
        else:
            pick.update(obs, info)

        if step % 20 == 0:
            print(
                f"step={step:03d}, "
                f"mode={'PLACE' if using_place else 'BC   '}, "
                f"phase={phase:22s}, "
                f"cube={np.array2string(info['cube_pos'], precision=3)}, "
                f"frame={np.array2string(info['frame_pos'], precision=3)}, "
                f"tcp={np.array2string(info['tcp_pos'], precision=3)}, "
                f"lifted={info['cube_lifted']}, "
                f"success={info['is_success']}"
            )

        if using_place and place.phase == "object_aware_done" and place.phase_steps >= 10:
            break

        if truncated:
            break

    final_cube = final_info["cube_pos"]
    final_frame = final_info["frame_pos"]

    xy_dist = float(np.linalg.norm(final_cube[:2] - final_frame[:2]))
    z_margin = float(final_cube[2] - final_frame[2])
    max_lift_delta = float(max_cube_z - initial_cube_z)

    final_success = bool(final_info["is_success"])

    print("最终 mode:", "PLACE" if using_place else "BC")
    print("最终 pick_phase:", pick.phase)
    print("最终 place_phase:", place.phase)
    print("最终 cube_pos:", np.array2string(final_cube, precision=4))
    print("最终 frame_pos:", np.array2string(final_frame, precision=4))
    print("final xy_dist:", round(xy_dist, 4))
    print("final z_margin:", round(z_margin, 4))
    print("max_lift_delta:", round(max_lift_delta, 4))
    print("lifted_ever:", lifted_ever)
    print("success_ever:", success_ever)
    print("final_success:", final_success)
    print("episode return:", total_reward)

    return {
        "final_success": final_success,
        "success_ever": success_ever,
        "lifted_ever": lifted_ever,
        "using_place": using_place,
        "place_phase": place.phase,
        "total_reward": total_reward,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("使用 device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    model, phase_names, obs_aug_mean, obs_aug_std, action_mean, action_std, env_config = load_policy(MODEL_PATH, device)

    env = make_env(env_config)

    final_success_count = 0
    success_ever_count = 0
    lifted_count = 0
    returns = []
    final_place_phases = []

    print("")
    print("=" * 60)
    print("Object-aware Hybrid BC-pick + Cube-target Rule-place 测试")
    print("=" * 60)
    print("BC 模型:", MODEL_PATH)
    print("NUM_EPISODES:", NUM_EPISODES)
    print("MAX_STEPS:", MAX_STEPS)
    print("CUBE_X_RANGE:", CUBE_X_RANGE)
    print("CUBE_Y_RANGE:", CUBE_Y_RANGE)
    print("=" * 60)

    for ep in range(1, NUM_EPISODES + 1):
        result = run_episode(
            env=env,
            ep=ep,
            model=model,
            phase_names=phase_names,
            obs_aug_mean=obs_aug_mean,
            obs_aug_std=obs_aug_std,
            action_mean=action_mean,
            action_std=action_std,
            device=device,
        )

        final_success_count += int(result["final_success"])
        success_ever_count += int(result["success_ever"])
        lifted_count += int(result["lifted_ever"])
        returns.append(result["total_reward"])
        final_place_phases.append(result["place_phase"])

    env.close()

    print("")
    print("=" * 60)
    print("Object-aware Hybrid 测试总结")
    print("=" * 60)
    print(f"final_success 成功次数：{final_success_count}/{NUM_EPISODES}")
    print(f"final_success 成功率：{final_success_count / NUM_EPISODES * 100:.1f}%")
    print(f"success_ever 次数：{success_ever_count}/{NUM_EPISODES}")
    print(f"success_ever 比例：{success_ever_count / NUM_EPISODES * 100:.1f}%")
    print(f"lifted_ever 次数：{lifted_count}/{NUM_EPISODES}")
    print(f"lifted_ever 比例：{lifted_count / NUM_EPISODES * 100:.1f}%")
    print(f"平均 return：{np.mean(returns):.3f}")

    print("")
    print("final place phase 统计：")
    for phase in sorted(set(final_place_phases)):
        print(f"{phase}: {final_place_phases.count(phase)}")


if __name__ == "__main__":
    main()
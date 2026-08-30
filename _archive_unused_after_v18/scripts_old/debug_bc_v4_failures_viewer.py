from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn

from openarm_pickplace_env import OpenArmPickPlaceEnv
import right_pick_place as pp


ROOT = Path(__file__).resolve().parents[1]

MODEL_CANDIDATES = [
    ROOT / "bc_policy_best_current.pt",
    ROOT / "bc_policy_phase_v4_wide_933.pt",
    ROOT / "bc_policy_phase_v4_done_967.pt",
    ROOT / "bc_policy_phase_v4.pt",
]

MAX_STEPS = 300
SLEEP = 0.03

# 重点看这些点
TEST_POINTS = [
    # 左臂扫描失败带
    (0.497, 0.050),
    (0.516, 0.050),
    (0.534, 0.050),
    (0.497, 0.075),
    (0.516, 0.075),
    (0.534, 0.075),

    # 对照成功点
    (0.516, 0.000),
    (0.516, -0.050),
]


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


def choose_model_path():
    for p in MODEL_CANDIDATES:
        if p.exists():
            return p

    raise FileNotFoundError(
        "找不到 BC v4 模型。请确认至少存在一个：\n"
        + "\n".join(str(p) for p in MODEL_CANDIDATES)
    )


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
        print(f"[PHASE] {self.phase} -> {next_phase}")
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


def load_policy(model_path, device):
    checkpoint = safe_torch_load(model_path, device)

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

    print("已加载模型:", model_path)
    print("best_epoch:", checkpoint.get("epoch"))
    print("best_val_loss:", checkpoint.get("best_val_loss"))
    print("val_mae:", checkpoint.get("val_mae"))
    print("env_config:", env_config)

    return model, phase_names, obs_aug_mean, obs_aug_std, action_mean, action_std, env_config


def policy_action(
    model,
    obs,
    phase,
    phase_names,
    obs_aug_mean,
    obs_aug_std,
    action_mean,
    action_std,
    device,
):
    p = phase_onehot(phase, phase_names)
    obs_aug = np.concatenate([obs.astype(np.float32), p], axis=0).astype(np.float32)

    obs_tensor = torch.from_numpy(obs_aug).float().to(device)
    obs_norm = (obs_tensor - obs_aug_mean) / obs_aug_std

    with torch.no_grad():
        action_norm = model(obs_norm.unsqueeze(0)).squeeze(0)
        action = action_norm * action_std + action_mean

    action = action.cpu().numpy().astype(np.float32)
    action = np.clip(action, -1.0, 1.0)

    return action


def make_env(x, y, env_config):
    env = OpenArmPickPlaceEnv(
        render_mode="human",
        randomize_cube=True,
        cube_x_range=(float(x), float(x)),
        cube_y_range=(float(y), float(y)),
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


def diagnose_failure(history, final_info):
    lifted_ever = any(h["cube_lifted"] for h in history)
    success_ever = any(h["is_success"] for h in history)

    max_cube_z = max(h["cube_pos"][2] for h in history)
    init_cube_z = history[0]["cube_pos"][2]
    max_lift_delta = max_cube_z - init_cube_z

    final_cube = final_info["cube_pos"]
    final_frame = final_info["frame_pos"]

    xy_dist = float(np.linalg.norm(final_cube[:2] - final_frame[:2]))
    z_margin = float(final_cube[2] - final_frame[2])

    print("")
    print("-" * 70)
    print("诊断结果")
    print("-" * 70)
    print("lifted_ever:", lifted_ever)
    print("success_ever:", success_ever)
    print("max_lift_delta:", round(float(max_lift_delta), 4))
    print("final cube:", np.array2string(final_cube, precision=4))
    print("final frame:", np.array2string(final_frame, precision=4))
    print("final xy_dist:", round(xy_dist, 4))
    print("final z_margin:", round(z_margin, 4))

    if not lifted_ever and max_lift_delta < 0.015:
        print("判断：抓取/夹持阶段失败。方块没有被稳定抬起。")
    elif lifted_ever and not success_ever and xy_dist > 0.045:
        print("判断：抓起来了，但放置 XY 没对准黑框。")
    elif lifted_ever and not success_ever and z_margin <= 0.015:
        print("判断：抓起来了，但最终高度/落点不满足成功判定。")
    elif success_ever:
        print("判断：过程中达到过成功条件。")
    else:
        print("判断：需要看 viewer，可能是碰撞/推方块/释放异常。")


def run_one_point(
    x,
    y,
    model,
    phase_names,
    obs_aug_mean,
    obs_aug_std,
    action_mean,
    action_std,
    env_config,
    device,
):
    print("")
    print("=" * 80)
    print(f"开始调试固定点：x={x:.3f}, y={y:.3f}")
    print("=" * 80)

    env = make_env(x, y, env_config)

    obs, info = env.reset(seed=92000 + int(x * 1000) * 10 + int((y + 1.0) * 1000))

    controller = PhaseController()
    controller.reset(info)

    history = []
    total_reward = 0.0

    print("初始 cube_pos:", np.array2string(info["cube_pos"], precision=4))
    print("初始 frame_pos:", np.array2string(info["frame_pos"], precision=4))
    print("初始 tcp_pos:", np.array2string(info["tcp_pos"], precision=4))
    print("")
    print("观察重点：")
    print("1. move_grasp 时夹爪是否对准方块中心")
    print("2. close_gripper 时方块是被夹住，还是被推走")
    print("3. lift_object 时方块有没有一起上升")
    print("4. move_release / open_release 时是否撞黑框")
    print("")

    for step in range(MAX_STEPS):
        phase = controller.phase

        action = policy_action(
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

        next_obs, reward, terminated, truncated, next_info = env.step(action)
        total_reward += float(reward)

        obs = next_obs
        info = next_info

        controller.update(obs, info)

        history.append(
            {
                "step": step,
                "phase": phase,
                "cube_pos": info["cube_pos"].copy(),
                "tcp_pos": info["tcp_pos"].copy(),
                "frame_pos": info["frame_pos"].copy(),
                "finger": float(obs[22]),
                "lifter": float(obs[23]),
                "cube_lifted": bool(info["cube_lifted"]),
                "is_success": bool(info["is_success"]),
                "action": action.copy(),
            }
        )

        if step % 5 == 0:
            print(
                f"step={step:03d} | "
                f"phase={phase:14s} | "
                f"cube={np.array2string(info['cube_pos'], precision=3)} | "
                f"tcp={np.array2string(info['tcp_pos'], precision=3)} | "
                f"finger={float(obs[22]):+.3f} | "
                f"lifter={float(obs[23]):+.3f} | "
                f"lifted={info['cube_lifted']} | "
                f"success={info['is_success']} | "
                f"action={np.array2string(action, precision=2, suppress_small=True)}"
            )

        time.sleep(SLEEP)

        if controller.phase == "done" and controller.phase_steps >= 20:
            print("controller done，结束本点。")
            break

        if truncated:
            print("env truncated，结束本点。")
            break

    diagnose_failure(history, info)

    print("")
    print("本点 total_reward:", total_reward)
    print("最终 phase:", controller.phase)
    print("")
    input("按回车继续下一个点...")

    env.close()


def main():
    model_path = choose_model_path()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("使用 device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    (
        model,
        phase_names,
        obs_aug_mean,
        obs_aug_std,
        action_mean,
        action_std,
        env_config,
    ) = load_policy(model_path, device)

    print("")
    print("=" * 80)
    print("BC v4 固定失败点 viewer 调试")
    print("=" * 80)
    print("模型:", model_path)
    print("测试点:", TEST_POINTS)
    print("=" * 80)

    for x, y in TEST_POINTS:
        run_one_point(
            x=x,
            y=y,
            model=model,
            phase_names=phase_names,
            obs_aug_mean=obs_aug_mean,
            obs_aug_std=obs_aug_std,
            action_mean=action_mean,
            action_std=action_std,
            env_config=env_config,
            device=device,
        )

    print("")
    print("=" * 80)
    print("全部固定点调试完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
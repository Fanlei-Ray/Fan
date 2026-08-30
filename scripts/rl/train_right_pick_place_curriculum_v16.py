from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from openarm_right_pick_place_env_v16 import OpenArmRightPickPlaceEnv, RightPickPlaceEnvConfig


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "rl_right_pick_place_v16_reward_fix"
V15_DIR = ROOT / "outputs" / "rl_right_pick_place_v15_curriculum"
V14_DIR = ROOT / "outputs" / "rl_right_pick_place_v14"


@dataclass
class CurriculumStage:
    name: str
    timesteps: int
    cube_x_range: Tuple[float, float]
    cube_y_range: Tuple[float, float]


def make_env(stage: CurriculumStage, max_steps: int, seed: int):
    def _factory():
        cfg = RightPickPlaceEnvConfig(
            randomize_cube=True,
            max_steps=max_steps,
            cube_x_range=stage.cube_x_range,
            cube_y_range=stage.cube_y_range,
            timeout_penalty=12.0,
            success_bonus=120.0,
        )
        env = OpenArmRightPickPlaceEnv(cfg)
        env.reset(seed=seed)
        return Monitor(env)
    return _factory


def create_vec_env(stage: CurriculumStage, max_steps: int, seed: int, vecnormalize_path: Optional[Path], training: bool):
    raw_env = DummyVecEnv([make_env(stage=stage, max_steps=max_steps, seed=seed)])

    if vecnormalize_path is not None and vecnormalize_path.exists():
        print(f"[V16] loading VecNormalize: {vecnormalize_path}")
        env = VecNormalize.load(str(vecnormalize_path), raw_env)
    else:
        print("[V16] creating new VecNormalize")
        env = VecNormalize(raw_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    env.training = bool(training)
    env.norm_reward = bool(training)
    return env


def stage_paths(stage_index: int, stage: CurriculumStage):
    stage_dir = OUT_DIR / f"stage_{stage_index:02d}_{stage.name}"
    return {
        "dir": stage_dir,
        "model": stage_dir / "ppo_model.zip",
        "vec": stage_dir / "vecnormalize.pkl",
        "best": stage_dir / "best_model",
        "eval": stage_dir / "eval_logs",
        "ckpt": stage_dir / "checkpoints",
    }


def train_stage(stage_index: int, stage: CurriculumStage, start_model: Path, start_vec: Optional[Path], max_steps: int, seed: int):
    paths = stage_paths(stage_index, stage)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90)
    print(f"[V16] CURRICULUM STAGE {stage_index}: {stage.name}")
    print("=" * 90)
    print("cube_x_range:", stage.cube_x_range)
    print("cube_y_range:", stage.cube_y_range)
    print("timesteps:", stage.timesteps)
    print("max_steps:", max_steps)
    print("start_model:", start_model)
    print("start_vec:", start_vec)
    print("output_dir:", paths["dir"])
    print("=" * 90)

    env = create_vec_env(stage=stage, max_steps=max_steps, seed=seed + stage_index, vecnormalize_path=start_vec, training=True)
    eval_env = create_vec_env(stage=stage, max_steps=max_steps, seed=seed + 1000 + stage_index, vecnormalize_path=start_vec, training=False)
    eval_env.norm_reward = False

    if not start_model.exists():
        raise FileNotFoundError(f"start model not found: {start_model}")

    print(f"[V16] loading model: {start_model}")
    model = PPO.load(str(start_model), env=env, device="cpu")

    checkpoint_cb = CheckpointCallback(
        save_freq=25_000,
        save_path=str(paths["ckpt"]),
        name_prefix=f"ppo_v16_{stage.name}",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(paths["best"]),
        log_path=str(paths["eval"]),
        eval_freq=20_000,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )

    model.learn(total_timesteps=stage.timesteps, callback=[checkpoint_cb, eval_cb], reset_num_timesteps=False)

    model.save(str(paths["model"]))
    env.save(str(paths["vec"]))
    print("[V16] saved model:", paths["model"])
    print("[V16] saved vecnormalize:", paths["vec"])

    env.close()
    eval_env.close()
    return paths["model"], paths["vec"]


def default_start_model() -> Path:
    candidates = [
        V15_DIR / "ppo_right_pick_place_curriculum_final.zip",
        V14_DIR / "ppo_right_fixed_success.zip",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[-1]


def default_start_vec() -> Optional[Path]:
    candidates = [
        V15_DIR / "vecnormalize_curriculum_final.pkl",
        V14_DIR / "vecnormalize_fixed_success.pkl",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="V16 reward-fixed curriculum PPO training for right-arm pick-place.")
    parser.add_argument("--start-model", type=str, default=str(default_start_model()))
    parser.add_argument("--start-vecnormalize", type=str, default=str(default_start_vec() or ""))
    parser.add_argument("--max-steps", type=int, default=450)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage1", type=int, default=140_000)
    parser.add_argument("--stage2", type=int, default=220_000)
    parser.add_argument("--stage3", type=int, default=320_000)
    parser.add_argument("--stage4", type=int, default=420_000)
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--skip-stage2", action="store_true")
    parser.add_argument("--skip-stage3", action="store_true")
    parser.add_argument("--skip-stage4", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    start_model = Path(args.start_model)
    start_vec = Path(args.start_vecnormalize) if args.start_vecnormalize else None

    if not start_model.exists():
        print("[V16] ERROR: cannot find start model:", start_model)
        print("[V16] Expected either V15 final model or V14 fixed success model.")
        raise SystemExit(2)

    if start_vec is not None and not start_vec.exists():
        print("[V16] WARNING: VecNormalize start file not found, starting new normalization:", start_vec)
        start_vec = None

    stages = []
    if not args.skip_stage1:
        stages.append(CurriculumStage("tiny_random", args.stage1, (0.512, 0.520), (0.046, 0.054)))
    if not args.skip_stage2:
        stages.append(CurriculumStage("near_fixed", args.stage2, (0.508, 0.524), (0.042, 0.058)))
    if not args.skip_stage3:
        stages.append(CurriculumStage("medium_range", args.stage3, (0.500, 0.535), (0.035, 0.070)))
    if not args.skip_stage4:
        stages.append(CurriculumStage("full_range", args.stage4, (0.490, 0.550), (0.020, 0.080)))

    manifest = {
        "out_dir": str(OUT_DIR),
        "start_model": str(start_model),
        "start_vecnormalize": str(start_vec) if start_vec is not None else None,
        "max_steps": args.max_steps,
        "reward_fix": "one-time lift/frame bonuses, timeout penalty, bounded progress rewards",
        "stages": [],
    }

    model_path = start_model
    vec_path = start_vec
    for idx, stage in enumerate(stages, start=1):
        model_path, vec_path = train_stage(idx, stage, model_path, vec_path, args.max_steps, args.seed)
        manifest["stages"].append({
            "index": idx,
            "name": stage.name,
            "timesteps": stage.timesteps,
            "cube_x_range": list(stage.cube_x_range),
            "cube_y_range": list(stage.cube_y_range),
            "model": str(model_path),
            "vecnormalize": str(vec_path),
        })

    final_model = OUT_DIR / "ppo_right_pick_place_v16_final.zip"
    final_vec = OUT_DIR / "vecnormalize_v16_final.pkl"
    shutil.copyfile(model_path, final_model)
    if vec_path is not None and vec_path.exists():
        shutil.copyfile(vec_path, final_vec)

    manifest["final_model"] = str(final_model)
    manifest["final_vecnormalize"] = str(final_vec)
    manifest_path = OUT_DIR / "curriculum_manifest_v16.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("[V16] CURRICULUM TRAINING FINISHED")
    print("final_model:", final_model)
    print("final_vecnormalize:", final_vec)
    print("manifest:", manifest_path)
    print("=" * 90)


if __name__ == "__main__":
    main()

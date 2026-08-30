# OpenArm V17: Hybrid FSM + BT + Right RL Adapter with Rule Fallback

V17 keeps the validated V13 hybrid task planner and adds deployable right-arm RL execution.

## What changed

- `BimanualTaskPlanner` now accepts `right_adapter_mode`:
  - `rule`: use validated `right_rule_pick_place.py` only.
  - `rl`: use trained PPO policy only.
  - `rl_fallback`: try PPO first; if RL fails/timeouts, automatically fall back to the validated rule expert.
- `RightRLPolicyAdapter` runs the trained V16 PPO policy directly inside the planner's MuJoCo simulation.
- `RightRLFallbackAdapter` makes RL safe for demos by guaranteeing a rule fallback.
- Logs are written to V17-specific filenames.

## Recommended commands

From project root:

```bat
python scripts\task_planner\run_hybrid_demo.py --right-only --right-adapter rl_fallback
```

Then run the normal two-task demo:

```bat
python scripts\task_planner\run_hybrid_demo.py --right-adapter rl_fallback
```

Queue demo:

```bat
python scripts\task_planner\run_hybrid_demo.py --queue-demo --right-adapter rl_fallback
```

Rule-only baseline remains available:

```bat
python scripts\task_planner\run_hybrid_demo.py --right-adapter rule
```

RL-only diagnostic:

```bat
python scripts\task_planner\run_hybrid_demo.py --right-only --right-adapter rl
```

## Required RL files

Default paths:

```text
outputs\rl_right_pick_place_v16_reward_fix\ppo_right_pick_place_v16_final.zip
outputs\rl_right_pick_place_v16_reward_fix\vecnormalize_v16_final.pkl
```

You can override paths:

```bat
python scripts\task_planner\run_hybrid_demo.py --right-only --right-adapter rl_fallback ^
  --rl-model outputs\rl_right_pick_place_v16_reward_fix\ppo_right_pick_place_v16_final.zip ^
  --rl-vecnormalize outputs\rl_right_pick_place_v16_reward_fix\vecnormalize_v16_final.pkl
```

## Outputs

```text
outputs\bimanual_task_planner_demo\task_log_v17_rl_fallback.csv
outputs\bimanual_task_planner_demo\behavior_tree_trace_v17_rl_fallback.csv
outputs\bimanual_task_planner_demo\fsm_trace_v17_rl_fallback.csv
outputs\bimanual_task_planner_demo\collision_log_v17_rl_fallback.csv
outputs\bimanual_task_planner_demo\hybrid_rl_fallback_report_v17.md
```

## Report phrasing

The planner supports multiple action backends. The right arm can execute a learned PPO policy; if the learned policy fails or times out, the planner automatically switches to the validated rule expert. This demonstrates RL deployment through a unified adapter interface while preserving task-level reliability.

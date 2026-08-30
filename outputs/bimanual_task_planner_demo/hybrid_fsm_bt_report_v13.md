# OpenArm V13 Hybrid FSM + Behavior Tree Demo

## Purpose

This demo combines a Behavior Tree task executor with a finite-state machine trace. The Behavior Tree organizes task behaviors, while the FSM records the lifecycle states of each task.

## Behavior Tree

```text
Sequence(HybridPickPlaceTask)
  -> InitTask
  -> PerceiveObject
  -> SelectArm
  -> SafetyCheck
  -> PlanPath
  -> PathSafe
  -> ParkInactiveArm
  -> PickAndPlaceWithRecovery
  -> VerifySuccess
```

## FSM States

```text
INIT -> PERCEIVE -> SELECT_ARM -> SAFETY_CHECK -> PLAN_PATH -> PARK_INACTIVE_ARM -> EXECUTE_PICK_PLACE -> VERIFY -> DONE/FAILED
```

## Results

Total place_success: 2/2

| Task | Arm | Pick | Place | Lift | XY | Z margin |
|---|---|---:|---:|---:|---:|---:|
| task_left_001 | left | True | True | 0.0250 | 0.0231 | 0.0299 |
| task_right_001 | right | True | True | 0.1003 | 0.0244 | 0.0299 |

## Generated Logs

- Task log: `E:\FL_Personal\openarm_mujoco-master\openarm_mujoco-master\outputs\bimanual_task_planner_demo\task_log_v13_hybrid.csv`
- Behavior-tree trace: `E:\FL_Personal\openarm_mujoco-master\openarm_mujoco-master\outputs\bimanual_task_planner_demo\behavior_tree_trace_v13_hybrid.csv`
- FSM trace: `E:\FL_Personal\openarm_mujoco-master\openarm_mujoco-master\outputs\bimanual_task_planner_demo\fsm_trace_v13_hybrid.csv`
- Collision log: `E:\FL_Personal\openarm_mujoco-master\openarm_mujoco-master\outputs\bimanual_task_planner_demo\collision_log_v13_hybrid.csv`

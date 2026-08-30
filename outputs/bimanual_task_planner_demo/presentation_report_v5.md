# OpenArm 双臂上层任务规划系统 Demo V5 汇报记录

## 一、Demo 目标

本 demo 展示一个统一的双臂上层任务规划系统。系统接收 pick-and-place 任务队列，自动读取任务中的物体位置，根据工作区规则选择左臂或右臂，并在一只机械臂执行时让另一只机械臂进入 park/hold-safe 状态，实现基础双臂避让和任务调度。

## 二、系统结构

- **任务层**：`PickPlaceTask(object=orange_cube, target=black_frame)`
- **规划层**：状态机 `INIT -> PERCEIVE -> SELECT_ARM -> SAFETY_CHECK -> PARK_INACTIVE_ARM -> EXECUTE_PICK_PLACE -> VERIFY -> DONE`
- **安全层**：`SafetyManager` 记录 inactive arm、TCP 距离、workspace_ok
- **动作层**：`RobotActionLibrary.pick_and_place(arm, object, target)` 统一封装左/右臂调用
- **恢复层**：支持 retry 和 fallback replan
- **日志层**：输出 task log、execution plan、summary JSON 和本 report

## 三、自动选臂规则

```text
if cube_y >= 0.020: use right arm
else: use left arm
```

## 四、执行计划

| Order | Task ID | Cube Position | Planned Arm | Inactive Arm | Description |
|---:|---|---|---|---|---|
| 1 | queue_left_001 | (0.516, 0.000, 1.050) | left | right | 队列任务 1：左臂 pick-and-place |
| 2 | queue_right_001 | (0.516, 0.050, 1.050) | right | left | 队列任务 2：右臂 pick-and-place |
| 3 | queue_left_002 | (0.516, 0.000, 1.050) | left | right | 队列任务 3：左臂再次执行 |
| 4 | queue_right_002 | (0.516, 0.050, 1.050) | right | left | 队列任务 4：右臂再次执行 |

## 五、执行结果

| Task ID | Arm | Pick Success | Place Success | Lift Delta | XY Dist | Z Margin | Message |
|---|---|---:|---:|---:|---:|---:|---|
| queue_left_001 | left | True | True | 0.0250 | 0.0231 | 0.0299 | left_rule IK adapter |
| queue_right_001 | right | True | True | 0.1003 | 0.0244 | 0.0299 | right_rule_pick_place.run_trial |
| queue_left_002 | left | True | True | 0.0250 | 0.0231 | 0.0299 | left_rule IK adapter |
| queue_right_002 | right | True | True | 0.1003 | 0.0244 | 0.0299 | right_rule_pick_place.run_trial |

## 六、结果汇总

- Total tasks: **4**
- Place success: **4/4**
- Success rate: **100.0%**

## 七、可展示亮点

1. 单一 planner 连续调度左右臂，不再分别运行左右臂脚本。
2. 系统根据物体位置自动选择机械臂。
3. active arm 执行时 inactive arm 自动进入安全等待状态。
4. 支持状态机日志、行为树视图、执行计划导出和 summary JSON。
5. 支持故障注入与 retry 恢复逻辑，能够展示上层智能调度雏形。

## 八、生成文件

- Task log: `E:\FL_Personal\openarm_mujoco-master\openarm_mujoco-master\outputs\bimanual_task_planner_demo\task_log_v5.csv`
- Execution plan: `E:\FL_Personal\openarm_mujoco-master\openarm_mujoco-master\outputs\bimanual_task_planner_demo\execution_plan_v5.csv`
- Summary JSON: `E:\FL_Personal\openarm_mujoco-master\openarm_mujoco-master\outputs\bimanual_task_planner_demo\task_summary_v5.json`
- State machine Mermaid: `E:\FL_Personal\openarm_mujoco-master\openarm_mujoco-master\outputs\bimanual_task_planner_demo\state_machine_v5.mmd`
- Runbook: `E:\FL_Personal\openarm_mujoco-master\openarm_mujoco-master\outputs\bimanual_task_planner_demo\demo_runbook_v5.txt`

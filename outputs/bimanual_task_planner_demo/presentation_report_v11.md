# OpenArm 双臂上层任务规划系统 Demo V11 汇报记录

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
if cube_y >= -0.100: use right arm
else: use left arm
```

## 四、执行计划

| Order | Task ID | Cube Position | Planned Arm | Inactive Arm | Description |
|---:|---|---|---|---|---|
| 1 | shuffle-L4-ycb_cup | (0.440, -0.090, 1.006) | right | left | upper computer selected cup | source=yolov8-seg:yolov8n-seg.pt:cup | right-arm exclusive zone |

## 五、路径规划输出

V11 会为每个任务生成 waypoint-based path plan，保存到：`E:\FL_Personal\openarm_mujoco-master\openarm_mujoco-master\outputs\bimanual_task_planner_demo\path_plan_v11.csv`。每个 waypoint 包含目标位置、motion_type、safety_role、expected_state 和 safe 标记。

## 六、执行结果

| Task ID | Arm | Pick Success | Place Success | Lift Delta | XY Dist | Z Margin | Message |
|---|---|---:|---:|---:|---:|---:|---|
| shuffle-L4-ycb_cup | right | True | True | 0.0670 | 0.0251 | 0.0104 | right_rule_pick_place.run_trial |

## 七、结果汇总

- Total tasks: **1**
- Place success: **1/1**
- Success rate: **100.0%**

## 八、可展示亮点

1. 单一 planner 连续调度左右臂，不再分别运行左右臂脚本。
2. 系统根据物体位置自动选择机械臂。
3. active arm 执行时 inactive arm 自动进入安全等待状态。
4. 支持状态机日志、行为树视图、执行计划、路径计划和 summary JSON。
5. 支持故障注入与 retry 恢复逻辑，能够展示上层智能调度雏形。

## 九、生成文件

- Task log: `E:\FL_Personal\openarm_mujoco-master\openarm_mujoco-master\outputs\bimanual_task_planner_demo\task_log_v11.csv`
- Execution plan: `E:\FL_Personal\openarm_mujoco-master\openarm_mujoco-master\outputs\bimanual_task_planner_demo\execution_plan_v11.csv`
- Path plan: `E:\FL_Personal\openarm_mujoco-master\openarm_mujoco-master\outputs\bimanual_task_planner_demo\path_plan_v11.csv`
- Collision log: `E:\FL_Personal\openarm_mujoco-master\openarm_mujoco-master\outputs\bimanual_task_planner_demo\collision_log_v11.csv`
- Summary JSON: `E:\FL_Personal\openarm_mujoco-master\openarm_mujoco-master\outputs\bimanual_task_planner_demo\task_summary_v11.json`
- State machine Mermaid: `E:\FL_Personal\openarm_mujoco-master\openarm_mujoco-master\outputs\bimanual_task_planner_demo\state_machine_v11.mmd`
- Runbook: `E:\FL_Personal\openarm_mujoco-master\openarm_mujoco-master\outputs\bimanual_task_planner_demo\demo_runbook_v11.txt`

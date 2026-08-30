# OpenArm V18.3 Vision Grasp Demo

## Purpose

This demo adds MuJoCo camera perception before task planning. The system renders an RGB image from an overhead camera, detects the orange cube by color segmentation, estimates its world position, and feeds the detected pose into the bimanual task planner.

## Pipeline

```text
Camera RGB render -> orange cube segmentation -> pixel center -> world pose estimate -> XY calibration -> select arm -> path planning -> collision check -> pick-and-place
```

## Vision Calibration

- Calibration source: `default_v18_2_offset`
- XY offset: `(-0.01000, -0.03200, +0.00000)`
- Calibration file: `E:\FL_Personal\openarm_mujoco-master\openarm_mujoco-master\outputs\bimanual_task_planner_demo\vision_calibration_v18_3.json`

## Results

Total place_success: 2/2

| Task | Arm | Pick | Place | Lift | XY | Z margin |
|---|---|---:|---:|---:|---:|---:|
| vision_left_001 | left | True | True | 0.0251 | 0.0225 | 0.0299 |
| vision_right_001 | right | True | True | 0.1002 | 0.0259 | 0.0299 |

## Generated Logs

- Vision detection log: `E:\FL_Personal\openarm_mujoco-master\openarm_mujoco-master\outputs\bimanual_task_planner_demo\vision_detection_log_v18.csv`
- Task log: `E:\FL_Personal\openarm_mujoco-master\openarm_mujoco-master\outputs\bimanual_task_planner_demo\task_log_v18_vision.csv`
- Path plan: `E:\FL_Personal\openarm_mujoco-master\openarm_mujoco-master\outputs\bimanual_task_planner_demo\path_plan_v18_vision.csv`
- Collision log: `E:\FL_Personal\openarm_mujoco-master\openarm_mujoco-master\outputs\bimanual_task_planner_demo\collision_log_v18_vision.csv`
- Debug images: `E:\FL_Personal\openarm_mujoco-master\openarm_mujoco-master\outputs\bimanual_task_planner_demo\vision_debug_v18`
- Grid test CSV: `E:\FL_Personal\openarm_mujoco-master\openarm_mujoco-master\outputs\bimanual_task_planner_demo\vision_grid_test_v18_3.csv`
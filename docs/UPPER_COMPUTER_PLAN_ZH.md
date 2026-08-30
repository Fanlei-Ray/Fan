# OpenArm 视觉联调上位机方案（课程项目）

更新时间：2026-08-26

## 1. 结论

官方 OpenArm 当前没有可直接交付的、完整的桌面 GUI 上位机。官方开源的是可供上位机调用的底层与中间层：

- 项目入口：[enactic/openarm](https://github.com/enactic/openarm)
- ROS 2 控制与 MoveIt 2：[enactic/openarm_ros2](https://github.com/enactic/openarm_ros2)
- CAN/CAN-FD 电机通信与诊断 CLI：[enactic/openarm_can](https://github.com/enactic/openarm_can)
- 主从臂遥操作：[enactic/openarm_teleop](https://github.com/enactic/openarm_teleop)
- Dora 控制节点：[enactic/dora-openarm](https://github.com/enactic/dora-openarm)
- 数据采集接口：[enactic/openarm_dataset](https://github.com/enactic/openarm_dataset)

因此，本课程的“上位机”应做成一个薄客户端：负责显示相机/识别结果、显示机械臂状态、下发经过确认的任务、急停/取消、记录日志；运动学、轨迹规划、控制器和 CAN 通信继续放在 ROS 2/机器人控制端。上位机不应直接逐帧写 CAN 电机指令。

本地项目并非从零开始，已有：

- `scripts/vision/vision_cube_detector.py`：MuJoCo RGB 颜色分割、像素反投影、标定偏移和检测日志。
- `scripts/task_planner/run_vision_grasp_demo.py`：视觉检测到任务规划的仿真闭环。
- `build_v19_full_package.py`：已经设计 ROS 2 迁移包、`/openarm/task_command` 和 `/openarm/task_status` 桥接。
- 同级目录已有 `openarm_v19_ros2_migration`，但它不在老师指定的当前项目根目录内；后续若采用，应先有选择地合并进本项目，不要维护两套真源。

## 2. 推荐总体架构

```text
视觉同学程序
  相机图像 + 检测结果(ObjectDetection)
                 |
                 v
上位机 GUI（你负责）
  图像/框/置信度/坐标显示
  任务确认、启动、暂停、取消、急停
  关节状态、控制器状态、任务状态、日志
                 |
                 v
ROS 2 任务桥/MoveIt 2（联调层）
  坐标变换 -> 工作空间检查 -> 选臂 -> IK -> 轨迹
                 |
                 v
ros2_control + OpenArm hardware interface
                 |
                 v
openarm_can -> SocketCAN/CAN-FD -> 电机
```

仿真阶段把最下方替换为 MuJoCo backend。GUI 和视觉协议不变，这样先做软件在环，再接真机。

## 3. 上位机第一版应该有的页面

建议用 Python + PySide6。课程项目开发快，和现有 Python/MuJoCo 代码兼容；若上位机运行在 Ubuntu ROS 2 主机，PySide6 进程可以直接使用 `rclpy`。若必须运行在 Windows，而 ROS 2/真机端在 Ubuntu，则在 Ubuntu 端增加 WebSocket/REST 网关，上位机不要直接依赖 Windows ROS 2 环境。

### 主监控页

- 相机画面及检测框。
- 物体类别、置信度、相机坐标、机器人基座坐标、时间戳。
- 选择的机械臂、当前任务状态、错误原因。
- “确认并执行”“取消任务”“回零”“急停”按钮。

### 机械臂状态页

- 7 个关节的位置、速度、力矩；夹爪位置。
- 控制器在线状态、轨迹 Action Server 状态。
- 真机时显示驱动器温度、CAN 在线状态；Dora 官方节点的状态字段包括 `qpos/qvel/qtorque/tmos/trotor`，可借鉴此数据模型。

### 标定与联调页

- 相机内参、相机到机器人基座的外参版本。
- 点击图像点/检测点后显示三维坐标。
- 保存标定结果、重投影误差和坐标变换检查。
- “仿真/真机”“dry-run/允许运动”必须醒目区分。

### 日志页

- 收到的视觉消息、下发任务、状态机变化、错误和耗时。
- 一键导出 JSONL/CSV，所有任务带唯一 `task_id`。

## 4. 建议先冻结的联调协议

不要一上来传“最终关节角”。视觉同学只提交观测结果，上位机/任务桥负责校验，规划端负责 IK 和轨迹。

视觉结果建议采用如下 JSON（可封装到 ROS 2 自定义消息，临时联调也可用 WebSocket）：

```json
{
  "schema_version": "1.0",
  "frame_id": "camera_color_optical_frame",
  "stamp_ns": 1720000000000000000,
  "detections": [
    {
      "id": "det-001",
      "class_name": "orange_cube",
      "confidence": 0.96,
      "bbox_xyxy": [120, 80, 220, 190],
      "position_m": [0.12, -0.04, 0.63],
      "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]
    }
  ]
}
```

上位机下发任务：

```json
{
  "schema_version": "1.0",
  "task_id": "pick-20260826-001",
  "command": "pick_place",
  "object_id": "det-001",
  "target_pose": {
    "frame_id": "base_link",
    "position_m": [0.516, 0.050, 1.050],
    "orientation_xyzw": [0.0, 1.0, 0.0, 0.0]
  },
  "requested_arm": "auto",
  "dry_run": true
}
```

任务状态：

```json
{
  "schema_version": "1.0",
  "task_id": "pick-20260826-001",
  "state": "PLANNING",
  "progress": 0.35,
  "selected_arm": "right",
  "error_code": null,
  "message": "IK solved"
}
```

状态机至少包括：`IDLE -> TARGET_READY -> CONFIRMED -> PLANNING -> EXECUTING -> SUCCEEDED/FAILED/CANCELLED/ESTOP`。超时、消息过期、坐标系未知、低置信度、越界都必须拒绝执行。

## 5. 必须向视觉/联调同学索取的内容

以下内容缺一部分时可以做 GUI 假数据演示，但不能完成真实联调：

1. 相机型号、驱动、分辨率、帧率，以及图像获取示例。
2. 相机内参文件（通常是 `camera_info.yaml`）和畸变模型。
3. 相机安装方式：眼在手上或固定相机；相机坐标系名称。
4. 相机到 `base_link` 的外参/TF 标定文件及标定方法、误差报告。
5. 识别程序仓库或至少可运行包、依赖文件、启动命令、模型权重和类别表。
6. 检测结果协议：topic/端口、消息类型、字段、单位、坐标轴方向、频率、时间戳、示例数据。
7. 输出到底是 2D 框、深度点、3D 中心还是完整 6D 位姿；只有 2D 框且没有深度/已知平面时，无法通用恢复三维抓取位姿。
8. 真机 ROS 2 版本、Ubuntu 版本、ROS_DOMAIN_ID、网络拓扑和启动顺序。
9. OpenArm 硬件版本（v1/v2、单臂/双臂）、关节名、零位、方向、限位、减速比、夹爪型号。
10. 控制器清单和准确接口：topic、service、action 名称，以及 `ros2 control list_controllers`、`ros2 action list -t`、`ros2 topic list -t` 输出。
11. CAN 适配器型号、接口名、CAN/CAN-FD 波特率及电机 ID 映射；急停的硬件实现。
12. 课程验收场景：物体、放置区、成功判据、最大延迟/误差、是否要求真机或仅联调演示。

优先让同学直接提供一份“可回放样例包”：10～30 秒相机数据/视频、对应检测 JSON/rosbag、标定文件。这样上位机无需等待真机即可开发。

## 6. 当前代码需要先修正的接口风险

本地 `build_v19_full_package.py` 中桥接器默认使用：

- `/left_arm_controller/follow_joint_trajectory`
- `/right_arm_controller/follow_joint_trajectory`

但当前官方 `openarm_ros2` 控制器配置名称是：

- `left_joint_trajectory_controller`
- `right_joint_trajectory_controller`
- 另有左右 `forward_position_controller`、`forward_velocity_controller` 和夹爪控制器。

因此不能直接假定本地 V19 的 action 名称正确。必须以真机运行时的 `ros2 action list -t` 为准，并放进 YAML 配置，禁止写死。

另外，本地 V19 桥接器目前只检查目标点是否落在简单长方体工作空间，并发布 `BRIDGE_READY`；注释里的 MoveIt 2 IK/轨迹仍是后续工作，不代表已经完成真机动作闭环。

## 7. 推荐实施顺序

### 阶段 A：协议和假数据（先完成）

1. 冻结上述三类消息：检测、任务、状态。
2. 建 PySide6 GUI，用本地 JSON 回放显示图像、框和状态。
3. 完成任务确认、取消、急停锁存、日志导出。
4. 所有控制先保持 `dry_run=true`。

### 阶段 B：接现有 MuJoCo

1. 将 `vision_cube_detector.py` 的 `CubeDetection` 转成统一检测协议。
2. 用 `run_vision_grasp_demo.py` 作为仿真 backend。
3. GUI 不直接操作 MuJoCo data；通过 backend/队列通信，避免 UI 线程卡死。
4. 验证低置信度、越界、超时、取消、重复 task_id 等异常路径。

### 阶段 C：接视觉同学程序

1. 先回放 rosbag/JSONL，再接实时流。
2. 检查时间戳和 TF；把相机坐标转换到 `base_link`。
3. 画面显示原始点与变换后的机器人坐标；记录标定版本。
4. 只有当深度/平面假设明确、重投影误差达标时才允许“确认执行”。

### 阶段 D：ROS 2 dry-run

1. 合并本地 V19 bridge 思路到当前项目。
2. 对齐官方 `openarm_ros2` 的 joint/controller 名称。
3. 接 `/joint_states`、FollowJointTrajectory action、controller_manager 状态。
4. 先验证 Action Server、取消和状态回传，不使能电机。

### 阶段 E：真机低风险联调

1. 硬急停在手，清空工作区，限速 10%。
2. 单臂、单关节、小于约 2°；验证方向、零位、限位。
3. 回零和无物体 pre-grasp。
4. 空载单臂抓取，再做双臂；RL 策略最后接入，且保留规则控制 fallback。

## 8. 你现在可以先做、不必等同学的工作

- GUI 静态布局、状态机、日志、配置加载。
- JSON/rosbag 回放模式和模拟检测数据。
- MuJoCo backend 适配。
- 任务 ID、超时、取消、急停锁存、权限分级。
- YAML 配置化的 topic/action/frame 名称。
- 自动生成“联调环境检查报告”：节点、topic、action、controller、TF 是否齐全。

必须等同学/老师提供或确认的，是相机与识别接口、标定与 TF、真机控制器实际名称、硬件版本和验收指标。

## 9. 第一版目录建议

```text
upper_computer/
  README.md
  requirements.txt
  config/default.yaml
  app.py
  ui/main_window.py
  models/messages.py
  transport/base.py
  transport/replay.py
  transport/ros2.py
  transport/websocket.py
  backend/mujoco_backend.py
  safety/validator.py
  logging/session_logger.py
  tests/test_messages.py
  tests/test_safety.py
  samples/detection.jsonl
```

建议先实现 `replay + mujoco` 两种模式；拿到同学接口后再选 `ros2` 或 `websocket`，避免现在猜协议造成返工。

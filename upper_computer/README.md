# OpenArm 课程项目上位机（MVP）

这是视觉—上位机—机械臂联调的第一版可运行骨架。目前默认使用检测结果回放，不会连接或驱动真实电机。

## 已实现

- Tkinter 桌面界面：检测框、目标信息、任务状态和联调日志。
- 检测消息解析及字段校验。
- 置信度、消息时效、坐标系、类别和工作空间安全检查。
- `IDLE/TARGET_READY/CONFIRMED/PLANNING/EXECUTING/SUCCEEDED/FAILED/CANCELLED/ESTOP` 状态机。
- 确认执行、取消、回零 dry-run、急停锁存及人工复位。
- JSONL 回放传输层和模拟任务状态，便于在视觉与 ROS 2 尚未就绪时开发。
- 会话日志写入本目录下的 `runtime/`。
- ROS 2、实时视觉和电机状态采用适配器边界，后续接入时不改 UI 业务逻辑。
- 电控 USART10 固定帧编解码与抗分片/噪声流式解析（当前只读）。
- RealSense + YOLOv8 视觉资料审查和接口配置（标定完成前禁止用于执行）。
- 兼容视觉同学当前 `robot_state` WebSocket 消息：实时显示 JPEG、类别、像素中心和深度。
- 实时视觉链路具备消息大小限制、有界队列、心跳、断线重连和 2.5 秒数据超时提示。
- 旧视觉协议没有 `base_link` 三维坐标、检测框和置信度，因此实时模式被强制设为只观察，不能下发运动。
- MuJoCo 多物体视觉桥：YOLOv8 优先检测，通用权重未识别的程序化模型由显式 RGB 后备检测补齐；每个检测框都会显示真实来源。
- 多目标指定抓取：可在上位机下拉框中选择橙色方块、蓝色零件、手机模型或鼠标模型，再由右臂放入黑色框架。
- 双臂避让：执行前把非工作左臂送入严格停放位，右臂独占演示工作区；碰撞监视器记录机械臂间和机械臂—环境接触。

## 启动

课程汇报建议直接双击 `run_course_demo.bat`，统一选择仿真桥、仿真上位机、真机视觉只观察、串口扫描和自动测试。目录索引见 `PROJECT_STRUCTURE_ZH.md`，汇报与演示说明见 `report/演示与汇报说明.md`。

双击：

```text
run_upper_computer.bat
```

视觉同学程序已经在同机 `8091` 启动时，可双击 `run_vision_monitor.bat`。若视觉机在另一台电脑，把 `config/legacy_vision_ws.json` 中的 `127.0.0.1` 改为视觉机 IP。此模式只能显示像素和深度，执行按钮保持锁定。

MuJoCo 视觉与上位机联调时，依次双击：

```text
run_mujoco_vision_bridge.bat
run_mujoco_upper_computer.bat
```

第一项打开 MuJoCo viewer、复用 V18.3 的 home/场景初始化并在本机 `8765` 发布虚拟相机检测，第二项打开对应上位机。在“指定抓取”中选目标后点击“确认执行”，系统会先把所选物品送入项目既有成果验证过的取件位，再调用 `BimanualTaskPlanner` 和右臂规则动作完成抓取放置；完整说明见 `docs/MUJOCO_UPPER_INTEGRATION_ZH.md`。

若要演示“官方真实物体扫描 + 官方 YOLOv8-Seg + 不经过固定取件位的视觉直抓”，直接双击：

```text
run_ycb_real_objects_demo.bat
```

该独立配置使用 YCB 芥末瓶、香蕉、苹果和杯子，严格 YOLO 模式不启用 RGB 颜色后备，动作速度档为 0.40、实时显示为 1.80 倍节拍。上位机提供“重新打乱物品”按钮，从 4 个已标定安全布局中随机选择不同布局并重新运行 YOLO；布局矩阵 16/16 全部抓放成功、危险碰撞 0，实际 viewer 的“打乱→重新识别→抓杯子→放框”也已返回 `SUCCEEDED`。证据与能力边界见 `docs/YCB_REAL_OBJECT_DEMO_ZH.md`。原有程序化多物体演示和脚本保留不变，便于对照及回退。

本项目已经在 `upper_computer/.venv/` 安装客户端所需的 `websockets` 与 `pyserial`，启动脚本会优先使用该 E 盘环境，不需要把依赖安装到 C 盘。

或在项目根目录执行：

```powershell
upper_computer\.venv\Scripts\python.exe upper_computer\run.py
```

运行测试：

```powershell
upper_computer\.venv\Scripts\python.exe -m unittest discover -s upper_computer\tests -v
```

## 当前限制

- `transport.mode` 当前为 `replay`，执行按钮只产生模拟状态，不会运动真机。
- `config/legacy_vision_ws.json` 使用 `legacy_vision_ws` 模式，但属于只观察适配层，同样不会运动真机。
- `config/mujoco_vision_ws.json` 可实时接收仿真图像和四类目标，并把所选任务送入已有 V18.3 抓取规划器。当前采用“多物体展示区 + 已验证安全取件位”，还不是任意桌面位姿的通用抓取器。
- `config/mujoco_ycb_yolov8_seg.json` 在已标定的小范围桌面工作区直接按视觉坐标抓取 YCB 真实扫描物体；它已覆盖四个固定散乱测试位，但仍不等于整张桌面任意位置或任意 6D 姿态。
- 自带 `yolov8n.pt` 是通用 COCO 权重，能真实加载和推理，但不保证识别 MuJoCo 的简单几何模型；界面会明确标注 `yolov8:...` 或 `rgb_fallback:...`，不会把颜色检测冒充 YOLO。
- 样例检测位置已经是 `base_link` 坐标；真实视觉通常输出相机坐标，必须由视觉/ROS 2 节点通过已标定 TF 转为 `base_link`。
- ROS 2 controller/action 名称、电机状态字段和硬急停输入尚待联调同学提供。
- 软件“急停”不能替代物理急停按钮和断电回路。

## 后续接入位置

- 真实视觉：实现 `src/openarm_upper/transports/base.py` 的 `Transport` 接口。
- ROS 2：增加 `Ros2Transport`，订阅检测、`/joint_states` 和任务状态，使用 Action 客户端发送/取消轨迹或任务。
- MuJoCo：把现有 `scripts/vision/vision_cube_detector.py` 的输出转换为 `DetectionBatch`。
- 电控：只将状态映射到 `MotorHealth`；实时控制继续由 `ros2_control/openarm_can` 完成。

所有可变名称、限制和待确认字段位于 `config/default.json`，不要在 UI 内写死。

可直接发给同学填写的清单在 `docs/TEAMMATE_INTERFACE_CHECKLIST_ZH.md`，机器可读的空白模板在 `config/site_interface.template.json`。

电控资料审查结论在 `docs/ELECTRICAL_INTEGRATION_REVIEW_ZH.md`。当前固件只回传 J1，且上位机命令未接入电机控制，所以串口写入保持禁用。离线协议工具：

```powershell
E:\Anaconda\envs\openarm\python.exe upper_computer\tools\usart10_protocol_cli.py --help
```

电控已确认 3.3V TTL、115200、J1–J7 CAN ID/三路 CAN 路由和面对输出轴的关节限位。当前本机只检测到 COM3–COM6 蓝牙虚拟串口，尚未插入 USB-UART；可先运行 `tools/list_serial_ports.py` 只读重新枚举。拿到适配器型号、TX/RX/GND 接线并出现 USB-UART 候选后，才可用 `tools/usart10_readonly_monitor.py` 只读监视 J1。所需 `pyserial` 已安装在项目内 `.venv`，两个工具都不会写串口。

视觉资料审查结论位于 `docs/VISION_INTEGRATION_REVIEW_ZH.md`。视觉程序当前输出的 `x/y` 是像素、`z` 才是米，不能直接作为机械臂三维目标。

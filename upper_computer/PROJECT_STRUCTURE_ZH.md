# OpenArm 上位机项目结构

本目录按“运行入口—配置—应用源码—测试—工具—资料—报告”组织。为保证已有脚本和导入路径稳定，本次整理不移动原文件，只增加统一索引和汇报目录。

```text
upper_computer/
├─ run_course_demo.bat              # 课程演示统一入口
├─ run_mujoco_vision_bridge.bat     # MuJoCo + 虚拟相机桥
├─ run_mujoco_upper_computer.bat    # 仿真上位机
├─ run_ycb_real_objects_demo.bat    # 官方 YCB + YOLOv8-Seg 一键演示
├─ run_ycb_real_objects_bridge.bat  # YCB 仿真、严格 YOLO 和实时桥
├─ run_ycb_real_objects_upper_computer.bat # YCB 专用上位机配置
├─ run_vision_monitor.bat           # 真机视觉只观察模式
├─ run_upper_computer.bat           # 离线回放模式
├─ scan_serial_ports.bat            # 只读枚举串口
├─ config/                          # 三种传输模式、电控与现场接口配置
├─ src/openarm_upper/               # GUI、消息、安全、状态机、协议和传输层
├─ tests/                           # 30 项自动测试
├─ tools/                           # USART10 和串口只读工具
├─ docs/                            # 仿真、视觉、电控和联调审查文档
├─ vendor/                          # 同学提供的原始参考资料归档
├─ runtime/                         # 上位机会话 JSONL 日志
├─ samples/                         # 离线检测回放数据
└─ report/                          # 课程汇报 PPT、演示说明、汇报证据和构建文件
```

项目根目录中与上位机直接相关的仿真桥：

```text
scripts/vision/mujoco_upper_bridge.py
scripts/vision/ycb_upper_bridge.py
scripts/vision/vision_cube_detector.py
outputs/upper_computer_mujoco_bridge/
outputs/ycb_real_objects/
```

## 四套演示边界

1. `replay`：不依赖视觉或电机，验证 UI、消息、安全校验、状态机和日志。
2. `mujoco_vision_ws`：虚拟相机 RGB → 检测 → WebSocket → 上位机 → 已有 V18.3 右臂动作；仿真可执行。
3. `mujoco_ycb_yolov8_seg`：官方 YCB 扫描物体 → 严格官方 YOLOv8-Seg → 标定桌面坐标 → 不经过固定装载位的直接抓放；当前受控场景四类回归 4/4，能力边界见 `docs/YCB_REAL_OBJECT_DEMO_ZH.md`。
4. `legacy_vision_ws`：接收视觉同学 RealSense/YOLO WebSocket；由于缺少 `base_link` 标定坐标，只观察、不下发真机运动。

真机串口侧保持只读：协议编码、流式解析、端口扫描和 J1 遥测监视工具已经完成，但固件尚未提供七关节可执行闭环与安全命令，因此禁止串口写控制。

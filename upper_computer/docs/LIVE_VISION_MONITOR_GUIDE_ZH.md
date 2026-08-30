# 实时视觉监视联调说明

## 当前能做什么

上位机已兼容视觉同学现有的 `robot_state` WebSocket JSON，可以：

- 显示 `frame_b64` 中的实时 JPEG 画面；
- 在画面上标出 `(x, y)` 像素中心；
- 显示目标类别和 `z` 轴向深度（米）；
- 显示连接、重连和消息超时状态；
- 将不含图片的大字段摘要写入 `upper_computer/runtime/` 日志；
- 在消息突发时只保留最新帧，避免界面越积越慢。

当前模式明确为“只观察”。现有消息里的 `x/y` 是图像像素，`z` 是相机轴向深度，不是机械臂 `base_link` 坐标。上位机不会把这些值作为机械臂运动目标，确认执行按钮保持禁用。

## 启动顺序

1. 视觉电脑连接 RealSense，并启动视觉同学的 `realsense_yolo_csv.py`。该程序当前监听 `0.0.0.0:8091`。
2. 如果视觉和上位机在同一台电脑，保持 `config/legacy_vision_ws.json` 的 `ws://127.0.0.1:8091`。
3. 如果在两台电脑，把上述地址改成视觉电脑的局域网 IP，例如 `ws://192.168.1.20:8091`，并确认两台电脑能互相访问 TCP 8091 端口。
4. 双击 `run_vision_monitor.bat`。

若界面显示“连接失败/正在重连”，先确认视觉程序是否已经运行、IP 是否正确、Windows 防火墙是否放行 8091。客户端会自动重试，不必反复重启上位机。

## 视觉组仍需补充的关键字段

要从“只观察”升级到可生成抓取目标，视觉消息至少需要：

- 唯一目标 ID、检测框 `bbox_xyxy`、置信度和类别；
- 相机帧时间戳、图像宽高和 `frame_id`；
- 使用 D435i 实际内参将 `(u, v, depth)` 反投影得到相机坐标三维点；
- 相机到机械臂基座的手眼标定结果，并将目标转换为 `base_link` 下的米制 XYZ；
- 多目标选择规则、目标丢失规则和标定误差测试结果。

建议最终直接输出 `config/vision_yolov8_realsense.json` 与 `samples/detections.jsonl` 所描述的结构，不再把像素 `x/y` 命名成空间坐标。

## 电控组仍需补充的关键字段

当前 USART10 文档和固件足够做报文编解码与 J1 只读监视，但不够安全地下发真机运动。仍需：

- USB-UART 型号、TTL/RS-232/RS-485 电平、接线、端口号和实际波特率；
- 命令帧是否真正接入电机控制的固件版本与实机验证记录；
- 七个关节的电机 ID/方向/零位/减速比/软硬限位映射；
- 全关节位置、速度、故障码、使能状态的回传格式和频率；
- 硬急停、掉线、超时、限位和通信校验失败时的安全行为；
- ROS 2 控制器/action/topic 名称，或明确由上位机直连串口的系统分工。

在以上信息与台架验证齐全前，串口仍保持只读，软件急停也不能替代物理断电急停。

## 本地验证命令

在项目根目录运行：

```powershell
upper_computer\.venv\Scripts\python.exe upper_computer\run.py --config upper_computer\config\legacy_vision_ws.json --self-check
upper_computer\.venv\Scripts\python.exe -m unittest discover -s upper_computer\tests -v
```

当前自动化测试覆盖视觉旧消息解析、非法图片/深度拒绝、有界队列、本地 WebSocket 端到端接收、检测消息、安全校验、任务状态机以及 USART10 编解码与流式重同步。

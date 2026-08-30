# 向视觉与电控同学索取的接口资料

把本表直接发给对应同学填写。未知项不要猜，填写“未知/待测”。拿到资料后再更新 `config/default.json`，不要直接改 UI 代码。

## 视觉同学填写

### 本次资料已确认

- RealSense 彩色/深度 640×480@30 FPS，深度对齐彩色。
- Ultralytics YOLOv8n，意图识别 `cup/bottle`，但当前代码实际发送所有类别。
- 视觉组报告环境为 Python 3.8.10、Ultralytics 8.4.31、PyTorch 2.4.1+cu121，初始 conf 建议 0.5。
- 计划/已有 D435i，坐标轴约定 X 右/Y 下/Z 前；实物型号、序列号和内参仍需现场读取。
- 视觉组接受约定 JSON，并建议视觉机作为 WebSocket 服务端；ACK、心跳和重连仍需落到代码并联测。
- WebSocket 服务端口 8091，另向硬编码的 172.16.25.178:9080 主动发送。
- 当前 `x/y` 为像素，`z` 为深度米值；没有完成三维反投影和基座坐标转换。

### 仍需填写

- 正式源仓库和版本/commit（当前只有压缩包）：`待提供`
- Python/C++ 与依赖版本：`部分已知；仍缺 OpenCV、pyrealsense2、websockets 版本及环境导出文件`
- 启动命令：`待提供`
- 相机型号：`待提供`
- 图像分辨率、帧率：`已知 640×480@30，待相机实测确认`
- 图像 topic/URL/端口：`待提供`
- 检测结果 topic/URL/端口：`待提供`
- 消息类型与一条完整样例：`待提供`
- 输出内容（2D 框/深度/3D 点/6D 位姿）：`待提供`
- 坐标单位：`待提供`
- `frame_id`：`待提供`
- 相机内参 `camera_info.yaml`：`待提供`
- 相机到 `base_link` 的外参/TF：`待提供`
- 标定方法、日期、平均/最大误差：`待提供`
- 模型权重、类别表、最低推荐置信度：`待提供`
- 可回放样例（rosbag/视频+JSONL）：`待提供`

## ROS 2/联调同学填写

- Ubuntu 版本：`待提供`
- ROS 2 发行版：`待提供`
- `ROS_DOMAIN_ID`：`待提供`
- 机器人 IP、上位机 IP、网络连接方式：`待提供`
- 完整启动命令及顺序：`待提供`
- `ros2 node list` 输出：`待提供`
- `ros2 topic list -t` 输出：`待提供`
- `ros2 action list -t` 输出：`待提供`
- `ros2 service list -t` 输出：`待提供`
- `ros2 control list_controllers` 输出：`待提供`
- `ros2 control list_hardware_interfaces` 输出：`待提供`
- TF 树 PDF或 `view_frames` 结果：`待提供`
- 左/右臂 FollowJointTrajectory action 名：`待提供`
- 夹爪 action/topic 名：`待提供`
- 任务状态和取消接口：`待提供`

## 电控同学填写

### 本次资料已确认（仍需真机复核）

- STM32H723 + FreeRTOS，USART10 为 115200/8N1、24 字节小端二进制帧。
- 7 个达妙电机及 J1-J7 的 CAN/Master ID 已从固件提取。
- 三路 Classic CAN 路由和固件关节限位已提取。
- 当前串口只回传 J1；接收的上位机命令没有接入电机执行。
- 电控同学确认 J1–J7 的 CAN ID 为 `0x01–0x07`，路由为 J1/J2→FDCAN3、J3/J4→FDCAN1、J5/J6/J7→FDCAN2。
- CAN 波特率确认为 1 Mbit/s；USART 为 115200、3.3V TTL。
- 面对电机输出轴观察的顺/逆时针限位已录入 `config/electrical_roboarm.json`。
- 减速比按电控要求不参与上位机换算。
- 本机当前只有 COM3–COM6 蓝牙虚拟串口，尚未检测到 USB-UART。

### 仍需填写

- OpenArm 硬件版本、单/双臂：`待提供`
- 关节电机型号和驱动器型号：`待提供`
- 关节名—电机 ID—CAN ID 映射表：`J1–J7 与 CAN ID 已知；仍缺 MuJoCo/URDF 关节名映射复核`
- 各关节减速比、零位、正方向：`减速比无需处理；仍缺机器人坐标正方向和机械零位/偏置`
- 软件/机械限位：`面对输出轴的方向端点已知；仍需确认这是软件限位还是机械极限，并进行低速实测`
- 最大速度、加速度、力矩、电流：`待提供`
- CAN 适配器型号与驱动：`待提供`
- Linux 接口名（例如 `can0`）：`不适用于当前 STM32 内部 FDCAN 路由；若以后使用 PC-CAN 仍需提供`
- Classic CAN 或 CAN-FD：`已确认固件为 Classic CAN`
- 仲裁波特率、数据波特率：`Classic CAN 1 Mbit/s；无独立数据相位`
- 控制模式和控制周期：`待提供`
- 指令超时/看门狗行为：`待提供`
- 使能、失能、清故障、回零顺序：`待提供`
- 位置/速度/力矩/电流/温度/故障码反馈定义：`待提供`
- 硬急停、接触器和断电回路说明：`待提供`
- 电源电压和电流限制：`待提供`
- 已验证的驱动配置和启动日志：`待提供`
- `ip -details link show can0` 输出：`待提供`

## 老师/组长确认

- 验收是仿真、软件在环还是真机：`待确认`
- 演示物体和放置区：`待确认`
- 成功判据与允许误差：`待确认`
- 延迟/帧率要求：`待确认`
- 是否要求上位机显示电机温度/故障：`待确认`
- 是否要求手动关节控制：`待确认`
- 是否要求保存视频、日志或实验报告：`待确认`

## 建议同学返回的目录

```text
team_interfaces/
  vision/
    README.md
    camera_info.yaml
    camera_to_base.yaml
    detection_sample.jsonl
    sample.mp4 或 sample.bag
  ros2/
    environment.txt
    nodes.txt
    topics.txt
    actions.txt
    services.txt
    controllers.txt
    hardware_interfaces.txt
  electrical/
    motor_mapping.xlsx 或 csv
    joint_limits.yaml
    can_config.txt
    startup.md
    estop_and_power.pdf
```

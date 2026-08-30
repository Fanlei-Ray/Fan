# 视觉资料接入审查：RealSense + YOLOv8

审查日期：2026-08-26

## 已收到并确认

- 单文件程序 `realsense_yolo_csv.py` 和权重 `yolov8n.pt`。
- RealSense 彩色流与深度流均为 640×480、30 FPS，深度对齐到彩色画面。
- 使用 Ultralytics YOLO，代码意图关注 `cup` 和 `bottle`，但当前实际遍历并输出所有模型类别。
- 本地 WebSocket 服务绑定 `0.0.0.0:8091`，约每 40 ms 推送一次全局 `robot_state`，其中包含 JPEG base64 画面。
- 每个检测还会新建连接发送到硬编码的 `ws://172.16.25.178:9080/ws/vision`。
- 程序保存 CSV 并显示彩色和深度窗口。
- 视觉组补充说明其环境为 Python 3.8.10、Ultralytics 8.4.31、PyTorch 2.4.1+cu121；初始置信度建议 0.5。
- 视觉组计划/已有 D435i，但具体实物、序列号仍需现场确认；建议坐标轴为光学坐标 X 右、Y 下、Z 前。
- 视觉组接受按本项目 JSON 方案调整，并建议视觉机作为 WebSocket 服务端。

注意：“建议 ACK/心跳/重连”和“多目标初版规则”是方案状态，不等于收到的程序已全部实现。源码中有 WebSocket 服务和单次外发 ACK，但没有应用层心跳、持久连接重连、限长发送队列或可供上位机使用的完整多目标数组。

## 当前坐标不能用于机械臂抓取

代码发送的：

```json
{"class":"bottle","x":320,"y":240,"z":0.63}
```

其中 `x/y` 是图像像素 `cx/cy`，只有 `z` 是深度米值；CSV 标题却写成 `X(m),Y(m),Z(m)`。这不是相机三维坐标，更不是机器人 `base_link` 坐标。

正确链路应为：

```text
像素中心/抓取点 (u,v) + 稳健深度
  -> 使用 RealSense 彩色相机内参反投影
相机光学坐标 [Xc,Yc,Zc]（米）
  -> 使用手眼/外参标定 T_base_camera
机器人基座坐标 [Xb,Yb,Zb]
  -> 工作空间、时效和置信度检查
规划/抓取
```

没有相机内参和 `camera -> base_link` 外参时，上位机必须拒绝执行。

## 代码问题和联调风险

1. WebSocket 状态没有 `schema_version`、时间戳、`frame_id`、置信度、bbox、检测 ID 或坐标单位。
2. 多目标时，全局状态最终保存“最后一个框”的类别和坐标；`find_success` 可能指向更早的另一个框，字段会互相不一致。
3. 每个检测、每一帧都新建线程和 WebSocket 连接并等待 ACK，目标多或网络异常时可能产生大量线程。
4. 读取 bbox 中心单个深度像素，没有处理 0 深度、边缘噪声、遮挡或异常值；建议使用中心邻域/目标区域中位数及有效深度比例。
5. 没有检测置信度阈值，也没有按目标类别过滤；误检会全部发送。
6. 服务绑定所有网卡且没有认证/TLS，局域网任意客户端都可能发送 `command`。
7. 接收端 IP `172.16.25.178` 写死；没有配置文件、重连退避、连接超时或发送队列。
8. 相机、WebSocket、模型在模块导入时立即启动，不便测试、配置和安全关闭。
9. 模型路径相对当前工作目录；从别的目录启动可能找不到 `yolov8n.pt`。
10. 没有 `requirements.txt`、Python/RealSense SDK/Ultralytics 版本、相机型号和启动说明。
11. `.pt` 权重包含 pickle 元数据，压缩包没有提供可信下载来源或版本；确认来源前不应加载执行。

## 要求视觉同学修改的最小输出

建议视觉端只输出相机坐标观测，由 Ubuntu ROS 2/联调层通过 TF 转到 `base_link`：

```json
{
  "schema_version": "1.0",
  "stamp_ns": 1720000000000000000,
  "frame_id": "camera_color_optical_frame",
  "image_size": [640, 480],
  "detections": [
    {
      "id": "det-000123",
      "class_name": "bottle",
      "confidence": 0.93,
      "bbox_xyxy": [220, 90, 360, 420],
      "pixel_center_uv": [290, 255],
      "camera_position_m": [0.04, 0.02, 0.63],
      "depth_valid_ratio": 0.91
    }
  ]
}
```

图像可暂时保留 JPEG base64，但长期建议使用 ROS Image、WebSocket 二进制帧或单独的视频流，避免 JSON base64 体积膨胀。

## 仍缺的资料

- RealSense 具体型号、序列号、安装位置和 USB 连接方式。
- 实际运行环境：操作系统、Python、`pyrealsense2`、OpenCV、Ultralytics、PyTorch、websockets 版本。
- `requirements.txt`/conda 环境文件和唯一启动命令。
- 相机内参或导出方法；相机坐标系名称和轴向约定。
- 相机到机械臂 `base_link` 的手眼/外参标定文件、标定方法、日期和误差。
- 模型权重来源、训练数据、类别表、置信度阈值及杯子/瓶子验证指标。
- 一段带时间同步的彩色＋深度录制及期望检测 JSON，供无相机回放测试。
- 上位机和视觉机的最终 IP、端口、谁做服务端、ACK/重连/心跳约定。
- 多物体选择规则、目标丢失处理、最大允许检测年龄和抓取点/姿态定义。

## 联调放行条件

### 可以立即做

- 用现有 `yolov8n.pt` 做杯子/瓶子 2D 检测可行性测试。
- 在视觉机上显示 RGB、Depth 和 bbox。
- 依据约定 JSON 开发 WebSocket 协议和上位机假数据适配。

### 只能显示，不能驱动机械臂

- 相机型号/序列号未确认。
- 只有像素中心和单点深度，未得到相机三维坐标。
- 尚无 `camera_color_optical_frame -> base_link` 标定。
- 没有带时间戳、置信度、bbox 和数组的稳定消息实现。

### 允许进入抓取 dry-run 前必须通过

1. 提交并锁定环境文件和启动命令。
2. 在真实 D435i 上读取内参并完成像素反投影。
3. 完成外参/手眼标定，记录平均和最大误差。
4. 提供彩色＋深度回放及期望 JSON，完成离线协议测试。
5. WebSocket 输出符合约定 schema，包含时间戳、frame_id、置信度、bbox、稳定 ID 和多目标数组。
6. 上位机把结果转换到 `base_link` 后通过工作空间、消息时效和目标丢失检查。
7. 先只显示和记录，再进行 MuJoCo/ROS 2 dry-run；最后才允许真机。

## 当前接入状态

参考程序和权重已保存到 `vendor/yolov8_reference/`，已生成 `config/vision_yolov8_realsense.json`。由于当前 E 盘 `openarm` 环境缺少 `pyrealsense2`、OpenCV、Ultralytics 和 websockets，且权重来源、相机和标定尚未确认，本次没有执行视觉程序或加载 `.pt`。

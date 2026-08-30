# 官方 YCB 真实物体 + YOLOv8-Seg 仿真上位机

## 这版解决了什么

- 桌面物体不再是程序化彩色模具，而是 YCB 官方真实物体扫描网格与实物纹理：芥末瓶、香蕉、苹果、杯子。
- 使用项目 E 盘内的官方 `yolov8n-seg.pt`，严格 YOLO 模式，不用颜色识别补漏。
- 修正 MuJoCo RGB 到 Ultralytics NumPy/BGR 的通道转换错误。
- YOLO 实例分割掩膜给出检测框、像素中心和图像平面主轴角；经过相机/类别中心偏置标定后输出 `base_link` 桌面坐标。
- 上位机选择目标后直接在当前散乱坐标抓取，不把物体搬到固定取件位，也不重置目标姿态。
- 双臂待机时向外避让；执行前右臂沿可见轨迹回到已验证 IK 起点，左臂保持严格停靠。
- 动作使用 `speed_scale=0.40`，主要运动段相对原始时长缩短 60%，仍保留必要的接触稳定步数；0.35 档杯子放置回归不稳定，因此没有继续盲目加速。
- viewer 使用独立 `MjData` 副本，物理线程只产出快照，主线程独占 MuJoCo/OpenGL 显示，修复了窗口中途退出并让动作停在半空的问题。
- 动作轨迹仍按 0.40 稳定档计算，显示采用 `1.80x` 分批节拍；下达任务前由严格 YOLO 锁定目标，运动期间只传实时视频与物理快照，结束后恢复 YOLO，避免 CPU 重复推理堵塞画面和 WebSocket。
- 上位机每 1.5 秒收到执行进度；若 bridge 真正断开，当前任务会自动进入 `FAILED`，不会永久卡在 `EXECUTING`。
- 上位机新增“重新打乱物品”按钮：仅在任务空闲且使用 YCB 仿真配置时可用。每次从 4 个已标定安全布局中随机选择一个与当前不同的布局，机械臂先回收、物体重新布置、严格 YOLO 再识别，之后可重新指定抓取。
- 这里的“随机”是有限安全布局随机，不是整张桌面任意坐标。每个布局都有显式的相机质心残差标定，代码不在运行时读取物体 body 坐标冒充视觉结果。

## 一键演示

双击：

```text
upper_computer\run_ycb_real_objects_demo.bat
```

也可以先后双击：

```text
upper_computer\run_ycb_real_objects_bridge.bat
upper_computer\run_ycb_real_objects_upper_computer.bat
```

桥接脚本会一直保留 MuJoCo viewer 和命令行窗口；即使异常退出，窗口也会停在 `pause`，不会再出现闪一下看不到报错的问题。

上位机目标列表中可选择：

- `bottle`：YCB 芥末瓶
- `banana`：YCB 香蕉
- `apple`：YCB 苹果
- `cup`：YCB 杯子

完成一次抓放后，点击“重新打乱物品”，等待状态从 `PLANNING → EXECUTING → SUCCEEDED` 且目标列表重新出现，再选择下一件物体并点击“确认并执行”。连续点击时不会重复当前布局。

## 已验证结果

回归命令：

```powershell
upper_computer\.venv\Scripts\python.exe upper_computer\tools\run_ycb_real_object_regression.py --confidence 0.06 --speed-scale 0.40 --output regression_viewer_fix_speed040.json
```

无窗口四物体证据：`outputs/ycb_real_objects/regression_viewer_fix_speed040.json`。

实际 viewer + WebSocket + 苹果抓放证据：`outputs/ycb_real_objects/viewer_thread_smoke_speed040_rate180.json`。该次窗口级测试从 `EXECUTING` 到 `SUCCEEDED` 约 4.1 秒，窗口存活到终态。

安全随机布局矩阵：`outputs/ycb_real_objects/shuffle_layout_matrix.json`。4 个布局 × 4 类物体共 16 组，严格 YOLO 检出 16/16、抓起 16/16、放框 16/16、危险碰撞 0。

实际窗口“打乱 → 重新识别 → 抓杯子 → 放框”证据：`outputs/ycb_real_objects/viewer_shuffle_then_cup_smoke.json`，最终状态为 `SUCCEEDED 100%`。

这一次受控场景回归得到：

- 官方 YOLOv8-Seg 检出：4/4
- 抓起成功：4/4
- 放入黑框：4/4
- 最终危险碰撞：四项均为 0
- 无固定装载位搬运：四项均为 `no_loading_station_teleport=true`
- 实际 MuJoCo 窗口抓放：`SUCCEEDED 100%`，物体到黑框中心 XY 距离约 `0.0129 m`
- 安全随机布局重复抓放：16/16 全部成功

## “任意位置、任意姿态”的准确边界

当前可以称为“在经过标定和回归的右臂桌面工作区内，直接抓取四类散乱放置的直立/平放物体”，不能称为整张桌面任意位置、任意六维姿态。

- 已验证桌面范围约为 `x=0.45~0.58 m`、`y=-0.09~0.05 m`，且目标之间要为夹爪进近留出间距。
- 圆形或直立近似对称物体对绕 Z 轴旋转不敏感；香蕉场景包含平面旋转，但当前仍使用按类别标定的抓取模板。
- YOLO 检测/分割只给类别、二维框和掩膜；它本身不提供物体完整 6D 位姿。
- 侧躺、倾斜、堆叠、严重遮挡等任意 6D 抓取，需要 RGB-D、相机到 `base_link` 的手眼标定、6D 位姿估计（例如 FoundationPose/MegaPose 类方案）、抓取候选生成和碰撞感知运动规划。
- 真机还要重新采集相机内参/外参、各类抓取高度和夹爪宽度，仿真标定常数不能直接复制到 RealSense。

因此，课堂演示时应说“完成了官方真实物体模型的严格 YOLO 识别、指定目标、视觉坐标直抓和四类闭环回归”，不要说“已经解决任意物体任意姿态抓取”。

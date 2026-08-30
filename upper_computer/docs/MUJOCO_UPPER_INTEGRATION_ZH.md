# MuJoCo 视觉与上位机联调

## 结论

可行，而且适合在真机资料和标定尚未齐全时先验证软件链路。当前已经接通：

```text
MuJoCo overhead camera RGB
  -> YOLOv8n 通用权重优先推理
  -> 未识别仿真类别的显式 RGB 后备检测
  -> 四类目标的像素中心/检测框/真实检测来源
  -> 相机射线反投影到 base_link
  -> WebSocket detection_batch
  -> 上位机实时画面、指定目标、安全校验和任务状态机
  -> task_command
  -> 所选物品进入 V18.3 已验证安全取件位
  -> 左臂严格停放 + 右臂独占执行
  -> 已有 V18.3 BimanualTaskPlanner / right rule adapter
  -> MuJoCo viewer 中实际抓取放置
```

场景包含橙色方块、蓝色零件、手机模型、鼠标模型和黑色接收框。检测程序读取 MuJoCo 渲染的 RGB 图像，不读取物体 body 位置作为正常检测结果。检测失败时发送空目标，不会把仿真真值冒充视觉结果。

默认 `auto` 模式会真实加载 `upper_computer/vendor/yolov8_reference/yolov8n.pt` 并先运行 YOLOv8。该权重是 COCO 通用权重，真实照片中的手机、鼠标可能被识别，但程序化 MuJoCo 几何与训练域差异很大，当前实测四类仿真物体均由 RGB 后备检测补齐。检测来源随消息送到界面，例如 `yolov8:yolov8n.pt:cell phone` 或 `rgb_fallback:magenta`，因此汇报时可以准确表述为“YOLOv8 已接入，仿真域使用可审计后备检测”，不能表述为“当前四类全部由 YOLO 识别”。若要全部由 YOLO 输出，需要视觉同学提供用仿真截图或实物数据训练的 `best.pt` 和类别表。

## 启动

在 `upper_computer` 目录按顺序双击：

1. `run_mujoco_vision_bridge.bat`
2. `run_mujoco_upper_computer.bat`

第一个脚本打开 MuJoCo viewer，并在 `ws://127.0.0.1:8765` 以 5 FPS 持续发布 640×480 JPEG 和检测结果；第二个脚本打开上位机。关闭 viewer 后桥接服务结束，上位机会显示断线并自动重连。5 FPS 是当前 640×480 软件渲染与仿真实时性的平衡值，画面会连续更新而不是只显示首尾帧。

动作执行期间也会持续推送中间画面。桥接层复用原 V18.3 动作每一步已有的 `viewer.sync()` 作为安全采样点：规划线程只复制一致的仿真状态，WebSocket 主线程把状态应用到一份独立的 `MjData` 后渲染。因此不是动作前后各截一张图，物理步进与 OpenGL 渲染也不会跨线程争用同一份状态。为避免网络或界面处理较慢时拖住仿真，待渲染状态只保留最新一份。

MuJoCo viewer 应持续显示，直到主动关闭窗口或在桥接命令窗口按 `Ctrl+C`。启动脚本会先检查 8765 端口；如果上一次桥接程序仍在运行，会显示占用进程 PID 并暂停，不再让 viewer 打开后立即闪退。遇到该提示时，应关闭之前的桥接命令窗口，不要同时启动两个桥接服务。

如不需要 viewer，可在项目根目录运行无窗口版本：

```powershell
upper_computer\.venv\Scripts\python.exe scripts\vision\mujoco_upper_bridge.py --host 127.0.0.1 --port 8765 --fps 5
```

## 当前可以调试

- 虚拟相机位置、视场角和画面；
- 颜色阈值、检测框、目标丢失；
- 像素反投影和 `base_link` 坐标；
- JSON 字段、时间戳、类别、置信度；
- WebSocket 连接、断线重连、消息超时和积压丢帧；
- 上位机安全工作空间检查、目标显示、日志和任务状态机；
- 不同目标位置下是否选择左臂或右臂的上层规则。

## 执行策略与当前边界

上位机“指定抓取”下拉框列出当前帧检测到的四类目标。选择目标并点击“确认执行”后，上位机发送带 `object_id` 和 `object_class` 的 `task_command`；桥接服务返回真实的 `PLANNING / EXECUTING / SUCCEEDED / FAILED`，并调用项目已有 `BimanualTaskPlanner(right_adapter_mode="rule")`。机械臂会在 viewer 中执行完整抓取放置，不是只改最终物体坐标。

已有右臂规则动作是调好的固定点技能，不是任意坐标控制器。为复用用户此前已经完成的成果，场景把未选物品放在夹爪进近包络之外的展示区；执行时保持双臂静止，先用可见的抬升—横移—下降动画把所选物品换入 V18 已验证点 `(0.516, 0.050, 1.050)m`，随后才启动真实抓放。这个“安全取件工位”设计允许四类物品共用已验证动作，但不应被描述为任意散乱位姿抓取。

初版多物体场景曾把蓝色零件放在取件位旁约 64 mm，张开的夹爪会在进近时扫到它，导致所有目标都被推走。现已把待选物体分离到 `y=-0.075m` 展示行，并统一四类物体的抓取碰撞盒、质量、惯量和摩擦参数。四类回归均能抓起并落入黑框。

双臂干涉采用三层约束处理：

1. 演示工作区固定由右臂独占，不在中心区动态切换左右臂。
2. 执行前把左臂送入严格停放姿态；实测停放后两 TCP 距离约 `0.564m`，高于配置阈值 `0.24m`。
3. MuJoCo contact 监视器在安全检查、停放后和动作后记录接触；回归中的最终危险碰撞数为 0。物体—桌面、夹爪—目标物、目标物—黑框属于任务允许接触，不计为双臂危险碰撞。

启动场景也直接复用 `prepare_scene_for_detection()`，包括 `core.load_home()`、position actuator 同步、已验证方块位置和稳定仿真步骤，不再从 MuJoCo 未初始化默认姿态开始。

现有 V18.3 动作内部是阻塞式执行，仿真中的取消/急停请求暂时不能在一条动作的中间立即打断，只会记录并在动作返回后处理。这不影响真机，因为此模式不会打开串口/CAN，但后续若要做完善的仿真急停，需要给原动作循环增加可取消检查点。

上位机桥接执行产生的计划、碰撞和任务结果单独写入 `outputs/upper_computer_mujoco_bridge/`，不再覆盖原 V18.3 演示证据目录。

## 仿真无法代替的真机测试

- D435i 实际内参、深度噪声和畸变；
- 真实光照、遮挡和 YOLO 模型误检；
- 相机到机械臂基座的手眼标定误差；
- 网络抖动、USB 掉线和真实相机帧延迟；
- 电机方向、零位、减速比、限位和物理急停；
- 抓取接触、摩擦和模型与实物之间的偏差。

因此推荐顺序是：先完成当前 MuJoCo 软件联调，再接视觉同学的 D435i 流，最后在限速、限位和物理急停齐全后接真机。

## 验证命令

```powershell
upper_computer\.venv\Scripts\python.exe upper_computer\run.py --config upper_computer\config\mujoco_vision_ws.json --self-check
upper_computer\.venv\Scripts\python.exe -m unittest discover -s upper_computer\tests -v
upper_computer\.venv\Scripts\python.exe upper_computer\tools\run_mujoco_multi_object_regression.py --objects orange_cube blue_part phone mouse --detector color --output multi_object_regression.json
upper_computer\.venv\Scripts\python.exe upper_computer\tools\run_mujoco_multi_object_regression.py --objects orange_cube --detector auto --output yolo_auto_regression.json
```

回归结果写入 `outputs/upper_computer_mujoco_bridge/`。`color` 命令验证四类物体的确定性完整抓放；`auto` 命令验证 YOLOv8 权重能加载和运行，并在通用权重不命中仿真几何时明确回退。

## 多物体回归记录（2026-08-29）

- 橙色方块、蓝色零件、手机模型、鼠标模型：4/4 均 `SUCCEEDED`。
- 四类均由右臂抓起并落入黑色框架，最终中心到黑框中心距离小于 55 mm，满足项目原判据。
- 左臂严格停放后两 TCP 距离约 0.564 m；动作检查未发现危险碰撞。
- `auto` 模式已验证 `YOLOv8(yolov8n.pt) + RGB fallback` 可运行；当前程序化物体的检测来源为显式 RGB 后备。

## 实时画面回归记录（2026-08-26）

- 完整抓放动作的执行区间抽样 6.06 秒：收到 51 帧，帧号从 1246 连续增长到 1296，51 个 JPEG 内容均不相同；证明上位机收到的是运动中间画面，而不是重复首帧或末帧。
- `home` 短动作回归：2.11 秒内收到 9 帧，帧号 100～108，9 个 JPEG 内容均不相同，状态为 `PLANNING -> SUCCEEDED`。
- 上位机自动测试在加入多物体检测用例后为 30 项，后续以最新测试命令输出为准。
- 回归结束后已关闭测试桥接进程，8765 端口未留下监听程序。

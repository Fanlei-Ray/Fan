# 电控资料接入审查：roboarm / USART10

审查日期：2026-08-26

## 已确认内容

- 主控工程为 STM32H723 + FreeRTOS，存在 USART10、3 路 FDCAN 和 7 个达妙电机。
- USART10：115200 bit/s、8N1、无流控、DMA 收发、空闲中断判帧。
- 每帧 24 字节，小端序：`int16 + 3*uint16 + 3*float32`，外加 2 字节头和 2 字节尾。
- PC 到机械臂：`VG ... NE`；机械臂到 PC：`GV ... EN`。
- FreeRTOS tick 为 1000 Hz，`vTaskDelay(10)` 对应约 10 ms，固件理论回传约 100 Hz。
- 固件内部有 7 个电机反馈，但 USART10 当前只回传 `DM_Data_t[0]`（J1）。
- 上位机命令解析结果只保存到 `pc_data`；没有代码把 `pc_data` 下发给任一电机。
- 当前 CAN 使用 Classic CAN；J1/J2 在 FDCAN3，J3/J4 在 FDCAN1，J5/J6/J7 在 FDCAN2。
- 电控同学确认 J1–J7 CAN ID 依次为 `0x01–0x07`，CAN 波特率为 1 Mbit/s；三路路由与固件一致。
- USART 电平确认为 3.3V TTL，波特率确认为 115200；减速比不参与上位机换算。
- 顺/逆时针限位均以“面对电机输出轴”为观察方向，原始方向端点和排序后的数值区间均已写入 `config/electrical_roboarm.json`。

## 电控同学追加确认的关节表

| 关节 | CAN ID | STM32 CAN | 顺时针端点 rad | 逆时针端点 rad | 数值安全区间 rad |
|---|---:|---|---:|---:|---:|
| J1 | 0x01 | FDCAN3 | -3.0 | 2.5 | [-3.0, 2.5] |
| J2 | 0x02 | FDCAN3 | 0.2 | -3.0 | [-3.0, 0.2] |
| J3 | 0x03 | FDCAN1 | -1.5 | 1.5 | [-1.5, 1.5] |
| J4 | 0x04 | FDCAN1 | 2.3 | 0.0 | [0.0, 2.3] |
| J5 | 0x05 | FDCAN2 | 1.5 | -1.5 | [-1.5, 1.5] |
| J6 | 0x06 | FDCAN2 | -0.25 | 0.8 | [-0.25, 0.8] |
| J7 | 0x07 | FDCAN2 | -1.5 | 1.5 | [-1.5, 1.5] |

方向端点不能直接等同于 MuJoCo/URDF 正负方向。例如 J2 的顺时针端点为正数而逆时针端点为负数，说明机械零位/符号映射仍需共同标定。当前配置同时保留方向端点和 `min/max`，避免丢失这一信息。

截图还显示 J1、J4、J5 的电机工具 P/V/T 范围互不相同；这些是电机工具显示范围，不是已经批准的机械臂安全速度/力矩限值，不能拿其中一组作为七关节公共限值。

## 不能据此开放真机控制的原因

0. 这是自定义 STM32H723 `roboarm` 固件；尚未证明其机械结构、关节顺序、零位和官方 `enactic/openarm` MuJoCo/URDF 完全一致。必须先完成“仿真模型—DH 参数—真机关节”映射确认。
1. 24 字节命令帧没有关节编号，无法表达 7 个关节的独立命令。
2. `State` 的命令语义未定义，位置/速度/力矩命令单位和范围也未定义。
3. 没有使能、失能、清故障、急停、回零、取消等命令定义。
4. 没有长度、CRC、序号、时间戳、确认应答或命令看门狗。
5. 串口协议只返回 J1，无法监控其余 6 个关节、温度或完整故障状态。
6. 电平虽已确认为 3.3V TTL，但 USB-UART 适配器型号、接插件和 TX/RX/GND 引脚定义仍未提供。

因此，上位机当前只实现离线编解码和捕获数据解析，不主动打开串口，也不发送控制帧。

## 固件实现风险（需反馈给电控同学）

- `USART10_Receive_IDLE()` 没有检查 `data_length == 24`，但会访问 `data_length-1/-2` 并固定读取到偏移 21；短帧可能越界或解析无效数据。
- 固件只检查接收缓冲区开头和最后两个字节，不支持噪声重同步、粘包或多帧解析，与协议文档建议的流式解析不一致。
- `HAL_UART_Transmit_DMA()` 的返回值没有检查；若 DMA 忙，帧可能静默丢失。
- USART 发送任务逐字段读取由 FDCAN 中断更新的 `DM_Data_t[0]`，没有原子快照；可能组成跨两次更新的“撕裂帧”。
- `PC_comm.c` 的 `extern DM_Data_t` 没有保留定义处的 `volatile` 限定，建议统一声明。
- Classic CAN 配置关闭自动重传，需要确认是否符合真实总线可靠性设计。
- 协议无版本字段；固件升级后上位机无法自动判断兼容性。

## 已完成的上位机接入

- `src/openarm_upper/protocols/usart10.py`：严格 24 字节编解码、范围/有限值校验、分片/噪声/粘包流式解析。
- `tools/usart10_protocol_cli.py`：离线编码、单帧解码和二进制捕获解析；不会访问串口。
- `tools/usart10_readonly_monitor.py`：显式指定端口后只读监视 J1；不包含串口写调用。项目 `.venv` 已安装 pyserial，仍需等待 USB-UART 型号和接线确认。
- `tools/list_serial_ports.py`：只读枚举 COM 端口并区分蓝牙虚拟串口与 USB-UART 候选，不会打开端口。
- `config/electrical_roboarm.json`：协议、电机映射、固件常量和缺失项。
- `vendor/roboarm_reference/`：从压缩包中提取的协议、核心配置和业务源码；未复制体积较大的 HAL/CMSIS、Keil 编译产物。

离线示例：

```powershell
E:\Anaconda\envs\openarm\python.exe upper_computer\tools\usart10_protocol_cli.py decode "47 56 01 00 00 00 00 00 00 00 00 00 80 3F 00 00 00 00 00 00 00 00 45 4E"
```

## 下一步需要电控同学补充/修改

建议先把协议升级为：固定头 + 协议版本 + 消息类型 + 序号 + payload 长度 + payload + CRC16。命令至少应有单独消息类型：心跳、状态查询、使能、失能、急停、清故障、回零、7 关节目标、夹爪目标、取消。遥测应覆盖 7 个关节并带时间戳、在线位图、温度和故障码。

在固件没有完成上述最小闭环、硬急停没有验证之前，上位机不得开放串口写控制。

还需要电控组和机械组共同补全：官方 MuJoCo/URDF 关节名、真机 J1-J7、机器人坐标正方向、机械零位和角度偏置。CAN ID、CAN 路由和面对输出轴的限位已经确认，但尚不足以确定仿真角度到电机角度的符号/零位映射。未验证前禁止把仿真关节角直接发送给该固件。

## 本机端口查询结果

2026-08-26 已直接查询本机串口。当前只检测到 `COM3–COM6`，硬件 ID 均为 `BTHENUM` 蓝牙虚拟串口，没有 USB-UART 候选。因此 `usart10.port` 保持 `null`，不能把 COM3–COM6 中任意一个猜作机械臂端口。插入 3.3V TTL 适配器后运行：

```powershell
upper_computer\.venv\Scripts\python.exe upper_computer\tools\list_serial_ports.py
```

## 来源完整性

- 协议 DOCX SHA-256：`77FDE1CE8447ECD34C3A5478C3AFA70F43B1110B99FDACCF72F1B7F0661283BC`
- `roboarm.zip` SHA-256：`E24FE4F62EEE974E98B4AB122FC4C8B63EC39DFF4AAFAB02BA9E5279055D23F8`
- J1 工具截图 SHA-256：`D66BC18F37869505D01CF01AFE705B32373CDA74E4ABAE2622356BA14EDCF7E5`
- J2 参数截图 SHA-256：`EE8EE54C7FB081C29AFF392C59FBF1C377114C443D9448E916D4D9FF346AA446`
- J4 工具截图 SHA-256：`9ACB2057CBBEBB629EC1126DDDFA756751BE9AF4173C03B92E954BE88313C550`
- J5 工具截图 SHA-256：`A091E86F0C0F10CB8F208A4640D3E32E2DDCCC0E1410AA709877EEBDFC0D6B14`
- DOCX 因本机缺少 LibreOffice 未完成页面渲染 QA；协议字段已通过 DOCX 结构、压缩包内 Markdown 和固件源码三方交叉验证。

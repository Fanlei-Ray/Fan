import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import csv
from datetime import datetime
import asyncio
import websockets
import json
import threading
import base64

# =======================
# YOLOv8 模型
# =======================
model = YOLO("yolov8n.pt")
TARGET_CLASSES = ["cup", "bottle"]

# =======================
# RealSense 配置
# =======================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
align_to = rs.stream.color
align = rs.align(align_to)
pipeline.start(config)
print("✅ RealSense 已启动")

# =======================
# CSV 文件
# =======================
csv_filename = f"object_coordinates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
csv_file = open(csv_filename, mode='w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Timestamp', 'Class', 'X(m)', 'Y(m)', 'Z(m)'])

# =======================
# 全局状态
# =======================
robot_state = {
    "command": "",
    "command_target": "",
    "detected": False,
    "find_success": False,
    "x": 0,
    "y": 0,
    "z": 0,
    "class": "",
    "frame_b64": ""
}

# =======================
# 控制端 WebSocket 服务（8091）
# =======================
connected_clients = set()

def parse_command(text):
    text = text.lower()
    if "瓶子" in text or "bottle" in text:
        return "bottle"
    elif "杯子" in text or "cup" in text:
        return "cup"
    return ""

async def ws_handler(websocket):
    connected_clients.add(websocket)
    print(f"✅ 控制端已连接: {websocket.remote_address}")
    try:
        async def receive():
            while True:
                try:
                    msg = await websocket.recv()
                    data = json.loads(msg)
                    if "command" in data:
                        cmd = data["command"]
                        robot_state["command"] = cmd
                        robot_state["command_target"] = parse_command(cmd)
                        print(f"🗣️ 收到指令：{cmd} → 目标：{robot_state['command_target']}")
                except Exception:
                    break

        async def send():
            while True:
                try:
                    await websocket.send(json.dumps(robot_state, ensure_ascii=False))
                    await asyncio.sleep(0.04)
                except Exception:
                    break

        await asyncio.gather(receive(), send())
    finally:
        connected_clients.remove(websocket)
        print(f"❌ 控制端断开: {websocket.remote_address}")

async def ws_server():
    async with websockets.serve(ws_handler, "0.0.0.0", 8091):
        print("✅ 控制端 WebSocket 服务启动: ws://0.0.0.0:8091")
        await asyncio.Future()

def start_ws_server():
    asyncio.run(ws_server())

threading.Thread(target=start_ws_server, daemon=True).start()

# =======================
# CSV/目标数据 WebSocket 客户端（发送到 Java 接收端 9080）
# =======================
CSV_SERVER_IP = "172.16.25.178"  # 改成实际接收端 IP
CSV_SERVER_PORT = 9080

async def send_target_data(data_json):
    uri = f"ws://{CSV_SERVER_IP}:{CSV_SERVER_PORT}/ws/vision"
    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps(data_json, ensure_ascii=False))
            ack = await websocket.recv()
            try:
                ack_data = json.loads(ack)
                print(f"📤 发送目标数据成功，收到 ACK: {ack_data}")
            except Exception:
                print(f"📤 发送目标数据成功，收到 ACK: {ack}")
    except Exception as e:
        print(f"❌ 发送目标数据失败: {e}")

def send_target_data_thread(data_json):
    threading.Thread(target=lambda: asyncio.run(send_target_data(data_json)), daemon=True).start()

# =======================
# 主循环
# =======================
# =======================
# 主循环（不限制类别）
# =======================
try:
    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        results = model.predict(color_image, verbose=False)[0]

        detected = False
        find_success = False
        cx = cy = z = 0
        target_class = ""
        need_find = robot_state["command_target"]

        for box, cls_id in zip(results.boxes.xyxy, results.boxes.cls):
            class_name = model.names[int(cls_id)]

            # ✅ 去掉类别限制，检测所有类别
            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            z = depth_frame.get_distance(cx, cy)
            target_class = class_name
            detected = True
            if need_find and class_name == need_find:
                find_success = True

            # 保存 CSV
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            csv_writer.writerow([timestamp, class_name, cx, cy, round(z,3)])
            csv_file.flush()

            # 发送 JSON 数据到接收端
            data_json = {
                "class": class_name,
                "x": cx,
                "y": cy,
                "z": round(z,3)
            }
            send_target_data_thread(data_json)

            # 绘制框
            color = (0, 255, 0) if find_success else (255, 0, 0)
            cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(color_image, f"{class_name} {z:.2f}m",
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

        # 更新全局状态
        robot_state.update({
            "detected": detected,
            "find_success": find_success,
            "x": cx,
            "y": cy,
            "z": round(z,3),
            "class": target_class
        })

        # 🔥 修复画面编码（确保 UI 能解码）
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        success, buffer = cv2.imencode('.jpg', color_image, encode_param)
        if success:
            b64_str = base64.b64encode(buffer).decode("utf-8")
            robot_state["frame_b64"] = b64_str

        # 显示窗口
        cv2.imshow("Color", color_image)
        cv2.imshow("Depth", cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    csv_file.close()
    print("✅ 程序已退出，CSV 文件保存完毕")
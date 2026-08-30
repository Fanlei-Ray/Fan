from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
import sys
import threading
import time
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "src"))

from openarm_upper.messages import TaskCommand
from openarm_upper.transports.mujoco_vision_ws import MujocoVisionWebSocketTransport


class MujocoVisionWebSocketE2ETests(unittest.TestCase):
    def test_receives_detection_sends_command_and_receives_status(self):
        import websockets

        ready = threading.Event()
        stop = threading.Event()
        port_holder = []
        received = []

        async def handler(websocket):
            message = {
                "type": "detection_batch",
                "payload": {
                    "frame_id": "base_link",
                    "stamp_ns": time.time_ns(),
                    "image_size": [640, 480],
                    "image_b64": base64.b64encode(b"jpeg-bytes").decode("ascii"),
                    "detections": [{
                        "id": "sim-orange-cube",
                        "class_name": "orange_cube",
                        "confidence": 0.99,
                        "bbox_xyxy": [280, 224, 311, 255],
                        "position_m": [0.52, 0.0, 1.025],
                        "orientation_xyzw": [0, 0, 0, 1],
                    }],
                },
            }
            await websocket.send(json.dumps(message))
            command = json.loads(await asyncio.wait_for(websocket.recv(), timeout=3.0))
            received.append(command)
            task_id = command["payload"]["task_id"]
            for state, progress in (
                ("PLANNING", 0.15),
                ("EXECUTING", 0.45),
                ("SUCCEEDED", 1.0),
            ):
                await websocket.send(json.dumps({
                    "type": "task_status",
                    "payload": {
                        "task_id": task_id,
                        "state": state,
                        "progress": progress,
                        "selected_arm": "left",
                        "message": "test",
                        "stamp_ns": time.time_ns(),
                    },
                }))
            while not stop.is_set():
                await asyncio.sleep(0.02)

        async def server_main():
            async with websockets.serve(handler, "127.0.0.1", 0) as server:
                port_holder.append(server.sockets[0].getsockname()[1])
                ready.set()
                await asyncio.to_thread(stop.wait)

        thread = threading.Thread(target=lambda: asyncio.run(server_main()), daemon=True)
        thread.start()
        self.assertTrue(ready.wait(2.0))

        batches = []
        statuses = []
        transport = MujocoVisionWebSocketTransport(
            f"ws://127.0.0.1:{port_holder[0]}", reconnect_max_sec=0.2
        )
        transport.set_callbacks(batches.append, statuses.append, lambda _: None)
        deadline = time.monotonic() + 3.0
        while not batches and time.monotonic() < deadline:
            transport.poll()
            time.sleep(0.02)

        self.assertTrue(batches)
        self.assertEqual(batches[0].detections[0].class_name, "orange_cube")
        self.assertEqual(batches[0].image_jpeg, b"jpeg-bytes")

        transport.submit_task(TaskCommand(
            "sim-task", "pick_place", "sim-orange-cube", "base_link",
            (0.52, 0.0, 1.025), (0.0, 0.0, 0.0, 1.0), dry_run=True,
        ))
        deadline = time.monotonic() + 2.5
        while len(statuses) < 3 and time.monotonic() < deadline:
            transport.poll()
            time.sleep(0.03)

        transport.close()
        stop.set()
        thread.join(timeout=2.0)
        self.assertEqual([item.state for item in statuses], [
            "PLANNING", "EXECUTING", "SUCCEEDED"
        ])
        self.assertEqual(received[0]["type"], "task_command")

    def test_connection_loss_finishes_active_task_as_failed(self):
        import websockets

        ready = threading.Event()
        stop = threading.Event()
        port_holder = []

        async def handler(websocket):
            command = json.loads(await asyncio.wait_for(websocket.recv(), timeout=3.0))
            task_id = command["payload"]["task_id"]
            await websocket.send(json.dumps({
                "type": "task_status",
                "payload": {
                    "task_id": task_id,
                    "state": "EXECUTING",
                    "progress": 0.45,
                    "selected_arm": "right",
                    "message": "test motion",
                    "stamp_ns": time.time_ns(),
                },
            }))
            # Returning closes the socket and simulates a crashed bridge.

        async def server_main():
            async with websockets.serve(handler, "127.0.0.1", 0) as server:
                port_holder.append(server.sockets[0].getsockname()[1])
                ready.set()
                await asyncio.to_thread(stop.wait)

        thread = threading.Thread(target=lambda: asyncio.run(server_main()), daemon=True)
        thread.start()
        self.assertTrue(ready.wait(2.0))

        statuses = []
        connections = []
        transport = MujocoVisionWebSocketTransport(
            f"ws://127.0.0.1:{port_holder[0]}", reconnect_max_sec=0.2
        )
        transport.set_connection_callback(
            lambda state, message: connections.append((state, message))
        )
        transport.set_callbacks(lambda _: None, statuses.append, lambda _: None)
        deadline = time.monotonic() + 2.0
        while not any(state == "CONNECTED" for state, _ in connections) and time.monotonic() < deadline:
            transport.poll()
            time.sleep(0.02)

        transport.submit_task(TaskCommand(
            "disconnect-task", "pick_place", "sim-ycb_apple", "base_link",
            (0.55, 0.0, 1.006), (0.0, 0.0, 0.0, 1.0), dry_run=True,
        ))
        deadline = time.monotonic() + 3.0
        while not any(item.state == "FAILED" for item in statuses) and time.monotonic() < deadline:
            transport.poll()
            time.sleep(0.02)

        transport.close()
        stop.set()
        thread.join(timeout=2.0)
        self.assertIn("EXECUTING", [item.state for item in statuses])
        failed = next(item for item in statuses if item.state == "FAILED")
        self.assertEqual(failed.error_code, "SIM_BRIDGE_CONNECTION_LOST")


if __name__ == "__main__":
    unittest.main()

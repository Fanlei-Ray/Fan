from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time
import unittest

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "src"))

from openarm_upper.transports.legacy_vision_ws import LegacyVisionWebSocketTransport


WEBSOCKETS_AVAILABLE = importlib.util.find_spec("websockets") is not None


@unittest.skipUnless(WEBSOCKETS_AVAILABLE, "optional websockets dependency not installed")
class LegacyVisionWebSocketE2ETests(unittest.TestCase):
    def test_receives_current_teammate_json(self):
        import websockets

        ready = threading.Event()
        stop = threading.Event()
        port_holder = []

        async def handler(websocket):
            payload = {
                "command": "找杯子",
                "command_target": "cup",
                "detected": True,
                "find_success": True,
                "x": 300,
                "y": 210,
                "z": 0.55,
                "class": "cup",
                "frame_b64": "",
            }
            await websocket.send(json.dumps(payload, ensure_ascii=False))
            while not stop.is_set():
                await asyncio.sleep(0.02)

        async def server_main():
            async with websockets.serve(handler, "127.0.0.1", 0) as server:
                port_holder.append(server.sockets[0].getsockname()[1])
                ready.set()
                await asyncio.to_thread(stop.wait)

        server_thread = threading.Thread(
            target=lambda: asyncio.run(server_main()), daemon=True
        )
        server_thread.start()
        self.assertTrue(ready.wait(2.0))

        observations = []
        connections = []
        transport = LegacyVisionWebSocketTransport(
            f"ws://127.0.0.1:{port_holder[0]}", reconnect_max_sec=0.2
        )
        transport.set_raw_vision_callback(observations.append)
        transport.set_connection_callback(
            lambda state, message: connections.append((state, message))
        )
        transport.set_callbacks(lambda _: None, lambda _: None, lambda _: None)
        deadline = time.monotonic() + 3.0
        while not observations and time.monotonic() < deadline:
            transport.poll()
            time.sleep(0.02)

        transport.close()
        stop.set()
        server_thread.join(timeout=2.0)
        self.assertTrue(observations)
        self.assertEqual(observations[0].class_name, "cup")
        self.assertEqual(observations[0].pixel_center_uv, (300, 210))
        self.assertTrue(any(state == "CONNECTED" for state, _ in connections))


if __name__ == "__main__":
    unittest.main()

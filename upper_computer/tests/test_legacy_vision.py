from __future__ import annotations

import base64
from pathlib import Path
import sys
import unittest

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "src"))

from openarm_upper.messages import LegacyVisionObservation
from openarm_upper.transports.legacy_vision_ws import LegacyVisionWebSocketTransport


class LegacyVisionMessageTests(unittest.TestCase):
    def test_parses_current_teammate_payload(self):
        observation = LegacyVisionObservation.from_legacy_dict({
            "command": "找瓶子",
            "command_target": "bottle",
            "detected": True,
            "find_success": True,
            "x": 321,
            "y": 242,
            "z": 0.63,
            "class": "bottle",
            "frame_b64": base64.b64encode(b"jpeg-placeholder").decode("ascii"),
        })
        self.assertEqual(observation.pixel_center_uv, (321, 242))
        self.assertAlmostEqual(observation.depth_m, 0.63)
        self.assertEqual(observation.image_jpeg, b"jpeg-placeholder")
        self.assertNotIn("image_jpeg", observation.to_log_dict())

    def test_rejects_invalid_depth(self):
        with self.assertRaises(ValueError):
            LegacyVisionObservation.from_legacy_dict({"z": -1})

    def test_rejects_invalid_base64(self):
        with self.assertRaises(ValueError):
            LegacyVisionObservation.from_legacy_dict({"z": 0.5, "frame_b64": "???"})


class LegacyVisionQueueTests(unittest.TestCase):
    def test_poll_delivers_observation_on_gui_thread(self):
        transport = LegacyVisionWebSocketTransport("ws://127.0.0.1:8091")
        received = []
        transport.set_raw_vision_callback(received.append)
        observation = LegacyVisionObservation.from_legacy_dict({
            "detected": True, "x": 1, "y": 2, "z": 0.3, "class": "cup"
        })
        transport._put_latest("raw_vision", observation)
        transport.poll()
        self.assertEqual(received, [observation])


if __name__ == "__main__":
    unittest.main()

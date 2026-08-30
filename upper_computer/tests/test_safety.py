from __future__ import annotations

import sys
from pathlib import Path
import time
import unittest

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "src"))

from openarm_upper.config import load_config
from openarm_upper.messages import DetectionBatch
from openarm_upper.safety import SafetyValidator
from openarm_upper.state_machine import InvalidTransition, TaskStateMachine


class SafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(APP_ROOT / "config" / "default.json")
        cls.validator = SafetyValidator(cls.config)

    def make_batch(self, confidence=0.9, position=(0.5, 0.0, 1.0), frame="base_link"):
        return DetectionBatch.from_dict({
            "frame_id": frame,
            "stamp_ns": time.time_ns(),
            "detections": [{
                "id": "d1", "class_name": "orange_cube", "confidence": confidence,
                "bbox_xyxy": [1, 2, 10, 20], "position_m": list(position)
            }],
        })

    def test_accepts_safe_detection(self):
        batch = self.make_batch()
        self.assertTrue(self.validator.validate(batch, batch.detections[0]).accepted)

    def test_rejects_low_confidence(self):
        batch = self.make_batch(confidence=0.2)
        self.assertFalse(self.validator.validate(batch, batch.detections[0]).accepted)

    def test_rejects_outside_workspace(self):
        batch = self.make_batch(position=(0.9, 0.0, 1.0))
        self.assertFalse(self.validator.validate(batch, batch.detections[0]).accepted)

    def test_rejects_wrong_frame(self):
        batch = self.make_batch(frame="camera_frame")
        self.assertFalse(self.validator.validate(batch, batch.detections[0]).accepted)


class StateMachineTests(unittest.TestCase):
    def test_estop_is_latched(self):
        machine = TaskStateMachine()
        machine.emergency_stop()
        with self.assertRaises(InvalidTransition):
            machine.transition("IDLE")
        machine.reset_estop()
        self.assertEqual(machine.state, "IDLE")

    def test_happy_path(self):
        machine = TaskStateMachine()
        for state in ("TARGET_READY", "CONFIRMED", "PLANNING", "EXECUTING", "SUCCEEDED"):
            machine.transition(state)
        self.assertEqual(machine.state, "SUCCEEDED")


if __name__ == "__main__":
    unittest.main()

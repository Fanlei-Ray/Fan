from __future__ import annotations

import sys
from pathlib import Path
import base64
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from openarm_upper.messages import DetectionBatch


class MessageTests(unittest.TestCase):
    def test_valid_batch(self):
        batch = DetectionBatch.from_dict({
            "frame_id": "base_link",
            "stamp_ns": 1,
            "detections": [{
                "id": "d1", "class_name": "orange_cube", "confidence": 0.9,
                "bbox_xyxy": [1, 2, 10, 20], "position_m": [0.5, 0.0, 1.0]
            }],
        })
        self.assertEqual(batch.detections[0].id, "d1")

    def test_rejects_invalid_bbox(self):
        with self.assertRaises(ValueError):
            DetectionBatch.from_dict({
                "frame_id": "base_link",
                "detections": [{
                    "id": "d1", "class_name": "cube", "confidence": 0.9,
                    "bbox_xyxy": [10, 2, 1, 20], "position_m": [0.5, 0.0, 1.0]
                }],
            })

    def test_canonical_batch_decodes_inline_jpeg(self):
        content = b"not-a-real-jpeg-but-valid-transport-bytes"
        batch = DetectionBatch.from_dict({
            "frame_id": "base_link",
            "stamp_ns": 1,
            "image_b64": base64.b64encode(content).decode("ascii"),
            "detections": [],
        })
        self.assertEqual(batch.image_jpeg, content)

    def test_canonical_batch_rejects_invalid_image_base64(self):
        with self.assertRaises(ValueError):
            DetectionBatch.from_dict({
                "frame_id": "base_link",
                "stamp_ns": 1,
                "image_b64": "%%%invalid%%%",
                "detections": [],
            })


if __name__ == "__main__":
    unittest.main()

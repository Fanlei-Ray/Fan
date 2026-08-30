from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from vision.vision_cube_detector import OrangeCubeDetector


class MujocoOrangeSegmentationTests(unittest.TestCase):
    def test_accepts_bright_mujoco_39_orange(self):
        detector = object.__new__(OrangeCubeDetector)
        rgb = np.zeros((3, 3, 3), dtype=np.uint8)
        rgb[1, 1] = [255, 229, 137]
        mask = detector.segment_orange(rgb)
        self.assertTrue(bool(mask[1, 1]))
        self.assertEqual(int(mask.sum()), 1)

    def test_rejects_white_background(self):
        detector = object.__new__(OrangeCubeDetector)
        rgb = np.full((3, 3, 3), 255, dtype=np.uint8)
        self.assertEqual(int(detector.segment_orange(rgb).sum()), 0)


if __name__ == "__main__":
    unittest.main()

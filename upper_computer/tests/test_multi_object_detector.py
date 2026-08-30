from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from vision.multi_object_detector import MultiObjectDetector


class MultiObjectColorMaskTests(unittest.TestCase):
    def test_four_simulation_colors_are_separable(self):
        samples = {
            "orange": [255, 180, 70],
            "cyan": [40, 200, 230],
            "magenta": [210, 30, 120],
            "green": [40, 190, 55],
        }
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        for index, (kind, color) in enumerate(samples.items()):
            rgb.fill(0)
            rgb[index // 2, index % 2] = color
            mask = MultiObjectDetector.color_mask(rgb, kind)
            self.assertEqual(int(mask.sum()), 1, kind)

    def test_component_box_ignores_blue_marker_outside_table_roi(self):
        mask = np.zeros((480, 640), dtype=bool)
        mask[30:200, 500:530] = True
        mask[190:220, 330:360] = True
        self.assertEqual(
            MultiObjectDetector._mask_box(mask),
            (330, 190, 359, 219),
        )


if __name__ == "__main__":
    unittest.main()

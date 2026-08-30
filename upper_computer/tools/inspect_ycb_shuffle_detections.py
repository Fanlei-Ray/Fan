from __future__ import annotations

"""Print strict-YOLO coordinate error for each candidate shuffle layout."""

import os
from pathlib import Path
import sys

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
WORK = ROOT / "upper_computer" / "work"
os.environ.setdefault("YOLO_CONFIG_DIR", str(WORK / "ultralytics"))
os.environ.setdefault("TORCH_HOME", str(WORK / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(WORK / "matplotlib"))

from vision.ycb_upper_bridge import (  # noqa: E402
    YCB_OBJECT_SPECS,
    YCB_SHUFFLE_LAYOUTS,
    YCB_XML,
    YCBUpperBridge,
    prepare_ycb_scene,
)


model = mujoco.MjModel.from_xml_path(str(YCB_XML))
data = mujoco.MjData(model)
bridge = YCBUpperBridge(model, data, fps=8, realtime=False, yolo_confidence=0.06)
for layout_index, layout in enumerate(YCB_SHUFFLE_LAYOUTS, start=1):
    prepare_ycb_scene(model, data, realtime=False, spawns=layout)
    bridge._current_layout_index = layout_index - 1
    bridge.build_message()
    print(f"layout {layout_index}")
    for spec in YCB_OBJECT_SPECS:
        detection = bridge.last_detections_by_id.get(f"sim-{spec.body_name}")
        if detection is None:
            print(f"  {spec.class_name}: NOT DETECTED")
            continue
        expected = layout[spec.body_name]
        error = float(np.linalg.norm(detection.stable_world_pos[:2] - expected[:2]))
        print(
            f"  {spec.class_name}: detected={detection.stable_world_pos[:2]} "
            f"expected={expected[:2]} error={error:.4f}m"
        )

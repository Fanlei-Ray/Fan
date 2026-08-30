from __future__ import annotations

"""Evaluate candidate inactive-left-arm park poses in the MuJoCo scene."""

from pathlib import Path
import sys

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from task_planner import core


CANDIDATES = (
    ("home", {}),
    ("out_a", {"left_joint1_ctrl": -0.60, "left_joint2_ctrl": -0.40, "left_joint4_ctrl": 1.20}),
    ("out_b", {"left_joint1_ctrl": -1.00, "left_joint2_ctrl": -0.50, "left_joint4_ctrl": 1.20}),
    ("out_c", {"left_joint1_ctrl": -1.20, "left_joint2_ctrl": -0.70, "left_joint4_ctrl": 1.00}),
    ("fold_a", {"left_joint1_ctrl": -0.80, "left_joint2_ctrl": -0.80, "left_joint3_ctrl": 0.40, "left_joint4_ctrl": 1.20}),
    ("fold_b", {"left_joint1_ctrl": -1.40, "left_joint2_ctrl": -0.60, "left_joint3_ctrl": 0.40, "left_joint4_ctrl": 1.00}),
)


def site(data: mujoco.MjData, model: mujoco.MjModel, name: str) -> np.ndarray:
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    return data.site_xpos[index].copy()


def main() -> None:
    xml_path = ROOT / "v2" / "demo_multi_object_vision.xml"
    for label, targets in CANDIDATES:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        core.load_home(model, data)
        for name, value in targets.items():
            core.set_ctrl(model, data, name, value)
        for _ in range(1200):
            mujoco.mj_step(model, data)
        left = site(data, model, "left_gripper_tcp")
        right = site(data, model, "right_gripper_tcp")
        print(
            f"{label:7s} left={np.array2string(left, precision=3)} "
            f"distance={np.linalg.norm(left - right):.3f}m contacts={data.ncon}"
        )


if __name__ == "__main__":
    main()

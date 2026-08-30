from __future__ import annotations

"""Headless regression for selectable MuJoCo multi-object pick/place tasks."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
from typing import Any

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

WORK = ROOT / "upper_computer" / "work"
os.environ.setdefault("YOLO_CONFIG_DIR", str(WORK / "ultralytics"))
os.environ.setdefault("TORCH_HOME", str(WORK / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(WORK / "matplotlib"))

from vision.mujoco_upper_bridge import (
    MujocoUpperBridge,
    VISION_XML,
    patch_bridge_output_paths,
    prepare_multi_object_scene,
)
from vision.vision_cube_detector import ensure_vision_xml
from vision.mujoco_upper_bridge import SOURCE_XML
import task_planner.core as core


async def run_object(object_id: str, detector_backend: str) -> dict[str, Any]:
    ensure_vision_xml(SOURCE_XML, VISION_XML)
    patch_bridge_output_paths()
    core.ensure_output_dir()
    model = mujoco.MjModel.from_xml_path(str(VISION_XML))
    data = mujoco.MjData(model)
    prepare_multi_object_scene(model, data, realtime=False)
    bridge = MujocoUpperBridge(
        model,
        data,
        fps=5,
        realtime=False,
        detector_backend=detector_backend,
    )
    bridge.build_message()
    detection = bridge.last_detections_by_id.get(object_id)
    if detection is None:
        return {"object_id": object_id, "state": "NOT_DETECTED"}

    statuses: list[dict[str, Any]] = []

    async def capture_status(
        task_id: str,
        state: str,
        progress: float,
        *,
        selected_arm: str | None = None,
        error_code: str | None = None,
        message: str = "",
    ) -> None:
        statuses.append(
            {
                "task_id": task_id,
                "state": state,
                "progress": progress,
                "selected_arm": selected_arm,
                "error_code": error_code,
                "message": message,
            }
        )

    bridge.send_status = capture_status  # type: ignore[method-assign]
    await bridge.execute_command(
        {
            "type": "task_command",
            "payload": {
                "task_id": f"regression-{object_id}",
                "command": "pick_place",
                "object_id": object_id,
                "object_class": detection.class_name,
                "frame_id": "base_link",
                "position_m": detection.stable_world_pos.tolist(),
                "requested_arm": "right",
            },
        }
    )
    body_pos = core.get_body_pos(model, data, detection.body_name)
    frame_pos = core.get_body_pos(model, data, "black_frame")
    return {
        "object_id": object_id,
        "class_name": detection.class_name,
        "source": detection.source,
        "final_state": statuses[-1]["state"] if statuses else "NO_STATUS",
        "selected_arm": statuses[-1].get("selected_arm") if statuses else None,
        "message": statuses[-1].get("message", "") if statuses else "",
        "xy_to_frame_m": float(np.linalg.norm(body_pos[:2] - frame_pos[:2])),
        "z_margin_m": float(body_pos[2] - frame_pos[2]),
        "detector_status": bridge.detector.status,
        **bridge.last_execution_metrics,
        "statuses": statuses,
    }


async def async_main(args: argparse.Namespace) -> None:
    core.RIGHT_BEST_CONFIG["pregrasp_offset"] = (
        np.asarray(core.RIGHT_BEST_CONFIG["pregrasp_offset"], dtype=float)
        + np.array([args.offset_x, args.offset_y, args.offset_z], dtype=float)
    )
    core.RIGHT_BEST_CONFIG["grasp_offset"] = (
        np.asarray(core.RIGHT_BEST_CONFIG["grasp_offset"], dtype=float)
        + np.array([args.offset_x, args.offset_y, args.offset_z], dtype=float)
    )
    object_ids = [f"sim-{name}" for name in args.objects]
    results = []
    for object_id in object_ids:
        print(f"\n[regression] {object_id}")
        results.append(await run_object(object_id, args.detector))
    output = ROOT / "outputs" / "upper_computer_mujoco_bridge" / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--objects",
        nargs="+",
        default=["orange_cube", "blue_part", "phone", "mouse"],
    )
    parser.add_argument("--detector", choices=("auto", "yolo", "color"), default="color")
    parser.add_argument("--offset-x", type=float, default=0.0)
    parser.add_argument("--offset-y", type=float, default=0.0)
    parser.add_argument("--offset-z", type=float, default=0.0)
    parser.add_argument("--output", default="multi_object_regression.json")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(async_main(parse_args()))

from __future__ import annotations

"""End-to-end regression for official YCB + strict YOLOv8-Seg execution."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time
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

from vision.ycb_upper_bridge import (  # noqa: E402
    YCB_XML,
    YCBUpperBridge,
    prepare_ycb_scene,
)
import task_planner.core as core  # noqa: E402


async def run_object(body_name: str, confidence: float, speed_scale: float) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(YCB_XML))
    data = mujoco.MjData(model)
    prepare_ycb_scene(model, data, realtime=False)
    bridge = YCBUpperBridge(
        model,
        data,
        fps=8,
        realtime=False,
        yolo_confidence=confidence,
        speed_scale=speed_scale,
    )
    bridge.build_message()
    object_id = f"sim-{body_name}"
    detection = bridge.last_detections_by_id.get(object_id)
    actual_initial = core.get_body_pos(model, data, body_name)
    if detection is None:
        return {
            "object_id": object_id,
            "final_state": "NOT_DETECTED",
            "actual_initial_xyz_m": actual_initial.tolist(),
            "detector_status": bridge.detector.status,
        }

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
                "progress": float(progress),
                "selected_arm": selected_arm,
                "error_code": error_code,
                "message": message,
            }
        )

    bridge.send_status = capture_status  # type: ignore[method-assign]
    started = time.perf_counter()
    await bridge.execute_command(
        {
            "type": "task_command",
            "payload": {
                "task_id": f"ycb-regression-{body_name}",
                "command": "pick_place",
                "object_id": object_id,
                "object_class": detection.class_name,
                "frame_id": "base_link",
                "position_m": detection.stable_world_pos.tolist(),
                "requested_arm": "right",
            },
        }
    )
    elapsed = time.perf_counter() - started
    final_pos = core.get_body_pos(model, data, body_name)
    frame_pos = core.get_body_pos(model, data, "black_frame")
    detection_error = float(
        np.linalg.norm(detection.stable_world_pos[:2] - actual_initial[:2])
    )
    return {
        "object_id": object_id,
        "class_name": detection.class_name,
        "confidence": float(detection.confidence),
        "source": detection.source,
        "image_angle_rad": float(detection.image_angle_rad),
        "detected_xyz_m": detection.stable_world_pos.tolist(),
        "actual_initial_xyz_m": actual_initial.tolist(),
        "detection_xy_error_m": detection_error,
        "no_loading_station_teleport": True,
        "speed_scale": speed_scale,
        "wall_time_s": elapsed,
        "final_state": statuses[-1]["state"] if statuses else "NO_STATUS",
        "xy_to_frame_m": float(np.linalg.norm(final_pos[:2] - frame_pos[:2])),
        "z_margin_m": float(final_pos[2] - frame_pos[2]),
        "detector_status": bridge.detector.status,
        **bridge.last_execution_metrics,
        "statuses": statuses,
    }


async def async_main(args: argparse.Namespace) -> None:
    results = []
    for body_name in args.objects:
        print(f"\n[YCB regression] {body_name}")
        result = await run_object(body_name, args.confidence, args.speed_scale)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    summary = {
        "scene": str(YCB_XML),
        "strict_official_yolov8_seg": True,
        "rgb_fallback": False,
        "direct_scatter_execution": True,
        "speed_scale": args.speed_scale,
        "results": results,
        "detected_count": sum(item.get("final_state") != "NOT_DETECTED" for item in results),
        "pick_success_count": sum(bool(item.get("pick_success")) for item in results),
        "place_success_count": sum(bool(item.get("place_success")) for item in results),
    }
    output = ROOT / "outputs" / "ycb_real_objects" / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--objects",
        nargs="+",
        default=["ycb_bottle", "ycb_banana", "ycb_apple", "ycb_cup"],
    )
    parser.add_argument("--confidence", type=float, default=0.08)
    parser.add_argument("--speed-scale", type=float, default=0.40)
    parser.add_argument("--output", default="regression.json")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(async_main(parse_args()))

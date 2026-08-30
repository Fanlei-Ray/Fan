from __future__ import annotations

"""Validate every supported object in every safe shuffle layout."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import mujoco


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

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


async def main(args: argparse.Namespace) -> None:
    model = mujoco.MjModel.from_xml_path(str(YCB_XML))
    data = mujoco.MjData(model)
    prepare_ycb_scene(model, data, realtime=False)
    bridge = YCBUpperBridge(
        model,
        data,
        fps=8,
        realtime=False,
        yolo_confidence=args.confidence,
        speed_scale=args.speed_scale,
    )
    results: list[dict[str, Any]] = []

    for layout_index, layout in enumerate(YCB_SHUFFLE_LAYOUTS):
        for spec in YCB_OBJECT_SPECS:
            prepare_ycb_scene(
                model,
                data,
                realtime=False,
                spawns=layout,
            )
            bridge._current_layout_index = layout_index
            bridge.build_message()
            object_id = f"sim-{spec.body_name}"
            detection = bridge.last_detections_by_id.get(object_id)
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
                statuses.append({
                    "task_id": task_id,
                    "state": state,
                    "progress": float(progress),
                    "selected_arm": selected_arm,
                    "error_code": error_code,
                    "message": message,
                })

            bridge.send_status = capture_status  # type: ignore[method-assign]
            started = time.perf_counter()
            if detection is not None:
                await bridge.execute_command({
                    "type": "task_command",
                    "payload": {
                        "task_id": f"shuffle-L{layout_index + 1}-{spec.body_name}",
                        "command": "pick_place",
                        "object_id": object_id,
                        "object_class": spec.class_name,
                        "frame_id": "base_link",
                        "position_m": detection.stable_world_pos.tolist(),
                        "requested_arm": "right",
                    },
                })
            metrics = dict(bridge.last_execution_metrics) if detection is not None else {}
            result = {
                "layout_index": layout_index,
                "body_name": spec.body_name,
                "class_name": spec.class_name,
                "spawn_xyz_m": layout[spec.body_name].tolist(),
                "detected": detection is not None,
                "detected_xyz_m": (
                    None if detection is None else detection.stable_world_pos.tolist()
                ),
                "detection_xy_error_m": (
                    None
                    if detection is None
                    else float(
                        ((detection.stable_world_pos[:2] - layout[spec.body_name][:2]) ** 2).sum()
                        ** 0.5
                    )
                ),
                "confidence": None if detection is None else float(detection.confidence),
                "source": None if detection is None else detection.source,
                "final_state": statuses[-1]["state"] if statuses else "NOT_DETECTED",
                "wall_time_s": time.perf_counter() - started,
                **metrics,
            }
            results.append(result)
            print(
                f"layout={layout_index + 1} {spec.class_name}: "
                f"detected={result['detected']} pick={result.get('pick_success')} "
                f"place={result.get('place_success')}"
            )

    summary = {
        "scene": str(YCB_XML),
        "safe_finite_shuffle_layouts": len(YCB_SHUFFLE_LAYOUTS),
        "objects_per_layout": len(YCB_OBJECT_SPECS),
        "total_cases": len(results),
        "strict_official_yolov8_seg": True,
        "speed_scale": args.speed_scale,
        "results": results,
        "detected_count": sum(bool(item["detected"]) for item in results),
        "pick_success_count": sum(bool(item.get("pick_success")) for item in results),
        "place_success_count": sum(bool(item.get("place_success")) for item in results),
        "dangerous_collision_count": sum(
            int(item.get("final_dangerous_collision_count", 0)) for item in results
        ),
    }
    output = ROOT / "outputs" / "ycb_real_objects" / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
    print(f"saved: {output}")
    if not (
        summary["detected_count"] == summary["total_cases"]
        and summary["pick_success_count"] == summary["total_cases"]
        and summary["place_success_count"] == summary["total_cases"]
        and summary["dangerous_collision_count"] == 0
    ):
        raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confidence", type=float, default=0.06)
    parser.add_argument("--speed-scale", type=float, default=0.40)
    parser.add_argument("--output", default="shuffle_layout_matrix.json")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))

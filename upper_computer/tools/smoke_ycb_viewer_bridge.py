from __future__ import annotations

"""Drive one real-time YCB bridge task and save a compact smoke-test report."""

import argparse
import asyncio
import json
from pathlib import Path
import time

import websockets


ROOT = Path(__file__).resolve().parents[2]


async def run(args: argparse.Namespace) -> None:
    task_id = f"viewer-smoke-{time.time_ns()}"
    statuses: list[dict] = []
    shuffle_statuses: list[dict] = []
    selected: dict | None = None
    async with websockets.connect(args.url, max_size=4_000_000) as websocket:
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            raw = json.loads(await asyncio.wait_for(websocket.recv(), timeout=15.0))
            if raw.get("type") != "detection_batch":
                continue
            detections = raw.get("payload", {}).get("detections", [])
            selected = next(
                (item for item in detections if item.get("class_name") == args.object_class),
                None,
            )
            if selected is not None:
                break
        if selected is None:
            raise RuntimeError(f"did not detect requested class: {args.object_class}")

        if args.shuffle_first:
            shuffle_task_id = f"shuffle-smoke-{time.time_ns()}"
            await websocket.send(json.dumps({
                "type": "task_command",
                "payload": {
                    "task_id": shuffle_task_id,
                    "command": "shuffle_scene",
                    "object_id": None,
                    "frame_id": "base_link",
                    "position_m": None,
                    "orientation_xyzw": None,
                    "requested_arm": "right",
                    "dry_run": False,
                },
            }, ensure_ascii=False))
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                raw = json.loads(await asyncio.wait_for(websocket.recv(), timeout=15.0))
                if raw.get("type") != "task_status":
                    continue
                status = raw.get("payload", {})
                if status.get("task_id") != shuffle_task_id:
                    continue
                shuffle_statuses.append(status)
                print(f"shuffle {status.get('state')} {status.get('message', '')}")
                if status.get("state") in {"SUCCEEDED", "FAILED", "CANCELLED"}:
                    break
            if not shuffle_statuses or shuffle_statuses[-1].get("state") != "SUCCEEDED":
                raise RuntimeError("safe shuffle did not succeed")

            selected = None
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                raw = json.loads(await asyncio.wait_for(websocket.recv(), timeout=15.0))
                if raw.get("type") != "detection_batch":
                    continue
                selected = next(
                    (
                        item
                        for item in raw.get("payload", {}).get("detections", [])
                        if item.get("class_name") == args.object_class
                    ),
                    None,
                )
                if selected is not None:
                    break
            if selected is None:
                raise RuntimeError("target was not detected after shuffle")

        await websocket.send(json.dumps({
            "type": "task_command",
            "payload": {
                "task_id": task_id,
                "command": "pick_place",
                "object_id": selected["id"],
                "object_class": selected["class_name"],
                "frame_id": "base_link",
                "position_m": selected["position_m"],
                "orientation_xyzw": selected.get(
                    "orientation_xyzw", [0.0, 0.0, 0.0, 1.0]
                ),
                "requested_arm": "right",
                "dry_run": False,
            },
        }, ensure_ascii=False))

        # Initial YOLO model warm-up has its own budget.  Give the actual
        # motion a fresh timeout instead of subtracting detector startup time.
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            raw = json.loads(await asyncio.wait_for(websocket.recv(), timeout=15.0))
            if raw.get("type") != "task_status":
                continue
            status = raw.get("payload", {})
            if status.get("task_id") != task_id:
                continue
            statuses.append(status)
            print(
                f"{status.get('state')} {float(status.get('progress', 0.0)):.0%} "
                f"{status.get('message', '')}"
            )
            if status.get("state") in {"SUCCEEDED", "FAILED", "CANCELLED"}:
                break

    final_state = statuses[-1]["state"] if statuses else "NO_STATUS"
    terminal_received = final_state in {"SUCCEEDED", "FAILED", "CANCELLED"}
    report = {
        "task_id": task_id,
        "requested_class": args.object_class,
        "selected_detection": {
            key: selected.get(key)
            for key in ("id", "class_name", "confidence", "position_m", "source")
        },
        "statuses": statuses,
        "shuffle_statuses": shuffle_statuses,
        "final_state": final_state,
        "viewer_process_survived_through_terminal_status": terminal_received,
    }
    output = ROOT / "outputs" / "ycb_real_objects" / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {output}")
    if not terminal_received:
        raise TimeoutError("bridge did not return a terminal task status")
    if final_state != "SUCCEEDED":
        raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8765")
    parser.add_argument("--object-class", default="apple")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--output", default="viewer_thread_smoke.json")
    parser.add_argument("--shuffle-first", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))

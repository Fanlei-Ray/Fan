from __future__ import annotations

"""Scan direct (non-staged) grasp positions for the official YCB objects.

This is a calibration utility, not the interactive demo.  Every trial starts
from the same robot home state, parks the non-target objects outside the work
area, and asks the existing right-arm rule controller to pick/place the target
at its actual table position.
"""

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import right_rule_pick_place as rr


SCENE = ROOT / "v2" / "demo_ycb_real_objects.xml"
OBJECTS = ("ycb_bottle", "ycb_banana", "ycb_apple", "ycb_cup")
PARK_POSITIONS = {
    "ycb_bottle": np.array([0.72, -0.22, 1.006]),
    "ycb_banana": np.array([0.76, -0.22, 1.006]),
    "ycb_apple": np.array([0.80, -0.22, 1.006]),
    "ycb_cup": np.array([0.84, -0.22, 1.006]),
}


def make_config(object_name: str, grasp_z: float) -> dict:
    grasp = np.array([-0.006, -0.020, grasp_z], dtype=float)
    return {
        "name": f"ycb_direct_{object_name}_z{grasp_z:.3f}",
        "site_type": "tcp",
        "pregrasp_offset": grasp + np.array([0.0, 0.0, 0.085]),
        "grasp_offset": grasp,
        "preplace_offset": np.array([0.0, 0.0, 0.120]),
        "place_offset": np.array([0.0, 0.0, 0.050]),
        "joint_biases": {"right_joint7_ctrl": -0.060},
    }


def run_one(
    object_name: str,
    x: float,
    y: float,
    grasp_z: float,
    speed_scale: float,
    post_release_retreat: bool,
) -> dict:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    rr.load_home(model, data)
    for parked_name, parked_pos in PARK_POSITIONS.items():
        if parked_name != object_name:
            rr.set_free_body_pos(model, data, parked_name, parked_pos)
    rr.FIXED_CUBE_POS = np.array([x, y, 1.006], dtype=float)
    result = rr.run_trial(
        model,
        rr.choose_right_site(model),
        make_config(object_name, grasp_z),
        do_place=True,
        realtime=False,
        data=data,
        object_name=object_name,
        target_name="black_frame",
        reset_home=False,
        speed_scale=speed_scale,
        post_release_retreat=post_release_retreat,
    )
    return {
        "object": object_name,
        "x": x,
        "y": y,
        "grasp_z": grasp_z,
        "speed_scale": speed_scale,
        "post_release_retreat": post_release_retreat,
        "pick_success": bool(result["pick_success"]),
        "place_success": bool(result["place_success"]),
        "lift_delta_m": float(result["lift_delta"]),
        "xy_to_frame_m": float(result["xy_dist"]),
        "z_margin_m": float(result["z_margin"]),
        "final_xyz_m": np.asarray(result["cube_final"], dtype=float).tolist(),
        "frame_xyz_m": np.asarray(result["frame_final"], dtype=float).tolist(),
        "final_minus_frame_xy_m": (
            np.asarray(result["cube_final"], dtype=float)[:2]
            - np.asarray(result["frame_final"], dtype=float)[:2]
        ).tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", choices=OBJECTS, default="ycb_sports_ball")
    parser.add_argument("--grasp-z", type=float, default=0.055)
    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument("--post-release-retreat", action="store_true")
    parser.add_argument("--points", nargs="+", default=["0.49,0", "0.52,0", "0.55,0", "0.49,-0.03", "0.52,-0.03", "0.55,-0.03"])
    parser.add_argument("--output", default="position_scan.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    for raw in args.points:
        x_text, y_text = raw.split(",", 1)
        result = run_one(
            args.object,
            float(x_text),
            float(y_text),
            args.grasp_z,
            args.speed_scale,
            args.post_release_retreat,
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))
    output = ROOT / "outputs" / "ycb_real_objects" / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()

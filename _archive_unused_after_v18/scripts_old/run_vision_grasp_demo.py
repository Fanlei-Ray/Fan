from __future__ import annotations

"""OpenArm V18: camera perception + planner grasp demo.

Run from project root:
    python scripts\task_planner\run_vision_grasp_demo.py
    python scripts\task_planner\run_vision_grasp_demo.py --both-arms
    python scripts\task_planner\run_vision_grasp_demo.py --right-adapter rl_fallback

What this demo adds on top of the V17 planner:
    1. Create/load v2/demo_vision.xml with an overhead MuJoCo camera.
    2. Render RGB from the camera.
    3. Segment the orange cube in the image.
    4. Estimate cube world x/y using camera back-projection.
    5. Feed the detected cube pose into the existing planner.
    6. Execute select-arm, path planning, collision checking and pick-place.
"""

from pathlib import Path
import argparse
import csv
import json
import sys
import time
from typing import Any, Dict, List

import mujoco
import mujoco.viewer
import numpy as np

TASK_PLANNER_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = TASK_PLANNER_DIR.parent
ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import task_planner.core as core
from vision.vision_cube_detector import (
    CAMERA_NAME,
    OrangeCubeDetector,
    append_detection_log,
    ensure_vision_xml,
    init_detection_log,
)


VISION_XML_PATH = ROOT / "v2" / "demo_vision.xml"
DETECTION_LOG_PATH = core.OUTPUT_DIR / "vision_detection_log_v18.csv"
VISION_REPORT_PATH = core.OUTPUT_DIR / "vision_grasp_report_v18.md"
VISION_DEBUG_DIR = core.OUTPUT_DIR / "vision_debug_v18"


def patch_v18_output_paths() -> None:
    core.LOG_PATH = core.OUTPUT_DIR / "task_log_v18_vision.csv"
    core.PLAN_PATH = core.OUTPUT_DIR / "execution_plan_v18_vision.csv"
    core.SUMMARY_PATH = core.OUTPUT_DIR / "task_summary_v18_vision.json"
    core.REPORT_PATH = core.OUTPUT_DIR / "presentation_report_v18_vision.md"
    core.STATE_MACHINE_PATH = core.OUTPUT_DIR / "state_machine_v18_vision.mmd"
    core.RUNBOOK_PATH = core.OUTPUT_DIR / "demo_runbook_v18_vision.txt"
    core.PATH_PLAN_PATH = core.OUTPUT_DIR / "path_plan_v18_vision.csv"
    core.COLLISION_LOG_PATH = core.OUTPUT_DIR / "collision_log_v18_vision.csv"


VISION_RIGHT_CASES = [
    {
        "task_id": "vision_right_001",
        "object_name": "orange_cube",
        "target_name": "black_frame",
        "cube_spawn_pos": np.array([0.516, 0.050, 1.050], dtype=float),
        "description": "相机识别右侧 orange_cube 后，planner 应选择右臂抓取。",
    }
]

VISION_BOTH_ARM_CASES = [
    {
        "task_id": "vision_left_001",
        "object_name": "orange_cube",
        "target_name": "black_frame",
        "cube_spawn_pos": np.array([0.516, 0.000, 1.050], dtype=float),
        "description": "相机识别中间/左侧 orange_cube 后，planner 应选择左臂抓取。",
    },
    {
        "task_id": "vision_right_001",
        "object_name": "orange_cube",
        "target_name": "black_frame",
        "cube_spawn_pos": np.array([0.516, 0.050, 1.050], dtype=float),
        "description": "相机识别右侧 orange_cube 后，planner 应选择右臂抓取。",
    },
]


def _set_free_body_pos(model, data, body_name: str, pos: np.ndarray) -> None:
    core.set_free_body_pos(model, data, body_name, np.asarray(pos, dtype=float))


def _try_set_ctrl(model, data, name: str, value: float) -> None:
    try:
        core.set_ctrl(model, data, name, value)
    except Exception:
        pass


def prepare_scene_for_detection(model, data, cube_spawn_pos: np.ndarray, viewer=None, realtime: bool = False) -> np.ndarray:
    """Place the cube, let it settle, and return the stable body position.

    The detector estimates the stable cube body center from the camera. For the
    planner/adapters we keep z at the spawn height 1.050, because the existing
    pick-and-place adapters expect that convention and then settle internally.
    """
    core.load_home(model, data)
    _set_free_body_pos(model, data, "orange_cube", cube_spawn_pos)
    _try_set_ctrl(model, data, "left_finger1_ctrl", 0.445)
    _try_set_ctrl(model, data, "right_finger1_ctrl", 0.445)
    _try_set_ctrl(model, data, "lifter_ctrl", 0.0)
    mujoco.mj_forward(model, data)
    if viewer is not None:
        viewer.sync()
    core.sim_steps(model, data, 700, viewer=viewer, realtime=realtime)
    stable = core.get_body_pos(model, data, "orange_cube")
    if viewer is not None:
        viewer.sync()
    return stable


def detect_case_to_task(model, data, detector: OrangeCubeDetector, case: Dict[str, Any], viewer=None, realtime: bool = False) -> core.PickPlaceTask:
    task_id = str(case["task_id"])
    cube_spawn = np.asarray(case["cube_spawn_pos"], dtype=float)
    print("\n" + "=" * 100)
    print(f"VISION PERCEPTION: {task_id}")
    print("planned spawn pos:", cube_spawn)
    print("=" * 100)

    stable = prepare_scene_for_detection(model, data, cube_spawn, viewer=viewer, realtime=realtime)
    print("[VISION] stable cube body pos before detection:", stable)

    det = detector.detect(task_id=task_id, planner_spawn_z=float(cube_spawn[2]), stable_cube_z=float(stable[2]))
    append_detection_log(DETECTION_LOG_PATH, task_id, det)

    print("[VISION] detected:", det.detected)
    print("[VISION] source:", det.source)
    print("[VISION] pixel_center:", det.pixel_center)
    print("[VISION] bbox:", det.bbox, "area:", det.area)
    print("[VISION] estimated stable_world_pos:", det.stable_world_pos)
    print("[VISION] planner_cube_pos:", det.planner_cube_pos)
    if det.debug_image_path:
        print("[VISION] debug image:", det.debug_image_path)

    return core.PickPlaceTask(
        task_id=task_id,
        object_name=str(case.get("object_name", "orange_cube")),
        target_name=str(case.get("target_name", "black_frame")),
        cube_pos=det.planner_cube_pos.copy(),
        description=str(case.get("description", "")) + " | cube_pos from camera vision",
    )


def write_vision_report(tasks: List[core.PickPlaceTask], results: List[core.TaskResult]) -> None:
    core.ensure_output_dir()
    success_count = sum(1 for r in results if r.place_success)
    lines: List[str] = []
    lines.append("# OpenArm V18 Vision Grasp Demo")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("This demo adds MuJoCo camera perception before task planning. The system renders an RGB image from an overhead camera, detects the orange cube by color segmentation, estimates its world position, and feeds the detected pose into the bimanual task planner.")
    lines.append("")
    lines.append("## Pipeline")
    lines.append("")
    lines.append("```text")
    lines.append("Camera RGB render -> orange cube segmentation -> pixel center -> world pose estimate -> select arm -> path planning -> collision check -> pick-and-place")
    lines.append("```")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(f"Total place_success: {success_count}/{len(results)}")
    lines.append("")
    lines.append("| Task | Arm | Pick | Place | Lift | XY | Z margin |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(f"| {r.task_id} | {r.selected_arm} | {r.pick_success} | {r.place_success} | {r.lift_delta:.4f} | {r.xy_dist:.4f} | {r.z_margin:.4f} |")
    lines.append("")
    lines.append("## Generated Logs")
    lines.append("")
    lines.append(f"- Vision detection log: `{DETECTION_LOG_PATH}`")
    lines.append(f"- Task log: `{core.LOG_PATH}`")
    lines.append(f"- Path plan: `{core.PATH_PLAN_PATH}`")
    lines.append(f"- Collision log: `{core.COLLISION_LOG_PATH}`")
    lines.append(f"- Debug images: `{VISION_DEBUG_DIR}`")
    VISION_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("[V18] vision report:", VISION_REPORT_PATH)


def print_v18_header(cases: List[Dict[str, Any]], right_adapter: str) -> None:
    print("=" * 100)
    print("OpenArm V18 Vision + Task Planner Demo")
    print("=" * 100)
    print("新增功能：MuJoCo 相机视觉识别 orange_cube，然后把检测位置交给 planner 抓取。")
    print("camera:", CAMERA_NAME)
    print("XML_PATH:", core.XML_PATH)
    print("VISION_XML_PATH:", VISION_XML_PATH)
    print("right_adapter:", right_adapter)
    print("cases:")
    for c in cases:
        p = c["cube_spawn_pos"]
        print(f"  {c['task_id']:18s} spawn=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) | {c.get('description', '')}")
    print("=" * 100)


def run(args) -> None:
    patch_v18_output_paths()
    ensure_vision_xml(core.XML_PATH, VISION_XML_PATH, camera_name=CAMERA_NAME)
    core.XML_PATH = VISION_XML_PATH
    core.ensure_output_dir()
    init_detection_log(DETECTION_LOG_PATH)

    cases = VISION_BOTH_ARM_CASES if args.both_arms else VISION_RIGHT_CASES
    print_v18_header(cases, right_adapter=args.right_adapter)

    model = mujoco.MjModel.from_xml_path(str(core.XML_PATH))
    data = mujoco.MjData(model)
    detector = OrangeCubeDetector(model, data, debug_dir=VISION_DEBUG_DIR)

    def build_tasks(viewer=None, realtime=False) -> List[core.PickPlaceTask]:
        tasks: List[core.PickPlaceTask] = []
        for case in cases:
            task = detect_case_to_task(model, data, detector, case, viewer=viewer, realtime=realtime)
            tasks.append(task)
        core.write_execution_plan(tasks)
        return tasks

    if args.no_viewer:
        tasks = build_tasks(viewer=None, realtime=False)
        planner = core.BimanualTaskPlanner(
            model,
            data,
            viewer=None,
            realtime=False,
            max_retries=args.max_retries,
            enable_fallback=not args.no_fallback,
            simulate_recovery=args.simulate_recovery,
            right_adapter_mode=args.right_adapter,
        )
        results = planner.run(tasks)
        write_vision_report(tasks, results)
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        print("viewer 已打开。2 秒后开始视觉识别 + 抓取 demo。")
        time.sleep(2.0)
        tasks = build_tasks(viewer=viewer, realtime=True)
        planner = core.BimanualTaskPlanner(
            model,
            data,
            viewer=viewer,
            realtime=True,
            max_retries=args.max_retries,
            enable_fallback=not args.no_fallback,
            simulate_recovery=args.simulate_recovery,
            right_adapter_mode=args.right_adapter,
        )
        results = planner.run(tasks)
        write_vision_report(tasks, results)

        if args.hold_viewer:
            print("viewer 保持运行。关闭 viewer 或 Ctrl+C 结束。")
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)


def parse_args():
    parser = argparse.ArgumentParser(description="OpenArm V18 vision perception + grasp demo")
    parser.add_argument("--both-arms", action="store_true", help="run left vision task and right vision task")
    parser.add_argument("--right-adapter", choices=["rule", "rl", "rl_fallback"], default="rule", help="right arm executor")
    parser.add_argument("--simulate-recovery", action="store_true", help="inject one failure before real action")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--no-hold-viewer", dest="hold_viewer", action="store_false")
    parser.set_defaults(hold_viewer=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()

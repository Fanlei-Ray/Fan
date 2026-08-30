from __future__ import annotations

"""OpenArm V18.3: visible camera + auto calibration + vision grasp demo.

Run from project root:
    python scripts\task_planner\run_vision_grasp_demo.py
    python scripts\task_planner\run_vision_grasp_demo.py --both-arms
    python scripts\task_planner\run_vision_grasp_demo.py --calibrate-vision --no-viewer
    python scripts\task_planner\run_vision_grasp_demo.py --vision-grid-test --no-viewer

What this demo adds on top of V18.2:
    1. Visible overhead camera marker remains available for presentation.
    2. Vision calibration JSON can be generated automatically from several
       known simulation positions.
    3. Detection logs now include raw camera pose, calibration offset, and
       calibrated planner pose.
    4. A no-grasp grid test can evaluate vision accuracy quickly.
"""

from pathlib import Path
import argparse
import csv
import json
import sys
import time
from typing import Any, Dict, List, Optional

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
    DEFAULT_VISION_XY_CALIBRATION_OFFSET,
    OrangeCubeDetector,
    append_detection_log,
    ensure_vision_xml,
    init_detection_log,
    load_calibration_offset,
    save_calibration_offset,
)


VISION_XML_PATH = ROOT / "v2" / "demo_vision.xml"
DETECTION_LOG_PATH = core.OUTPUT_DIR / "vision_detection_log_v18.csv"
VISION_REPORT_PATH = core.OUTPUT_DIR / "vision_grasp_report_v18.md"
VISION_DEBUG_DIR = core.OUTPUT_DIR / "vision_debug_v18"
CALIBRATION_PATH = core.OUTPUT_DIR / "vision_calibration_v18_3.json"
CALIBRATION_SAMPLES_CSV = core.OUTPUT_DIR / "vision_calibration_samples_v18_3.csv"
GRID_TEST_CSV = core.OUTPUT_DIR / "vision_grid_test_v18_3.csv"


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

# Calibration and accuracy-test points stay within the currently proven rule/IK
# grasp workspace. Calibration uses MuJoCo body pose as ground truth only for
# computing the camera offset; normal grasp still consumes the camera result.
VISION_CALIBRATION_CASES = [
    {"task_id": "calib_center", "cube_spawn_pos": np.array([0.516, 0.000, 1.050], dtype=float)},
    {"task_id": "calib_right_a", "cube_spawn_pos": np.array([0.516, 0.050, 1.050], dtype=float)},
    {"task_id": "calib_right_b", "cube_spawn_pos": np.array([0.526, 0.050, 1.050], dtype=float)},
    {"task_id": "calib_mid_a", "cube_spawn_pos": np.array([0.506, 0.030, 1.050], dtype=float)},
    {"task_id": "calib_mid_b", "cube_spawn_pos": np.array([0.526, 0.030, 1.050], dtype=float)},
]

VISION_GRID_TEST_CASES = [
    {"task_id": "grid_01", "cube_spawn_pos": np.array([0.506, 0.000, 1.050], dtype=float)},
    {"task_id": "grid_02", "cube_spawn_pos": np.array([0.516, 0.000, 1.050], dtype=float)},
    {"task_id": "grid_03", "cube_spawn_pos": np.array([0.526, 0.000, 1.050], dtype=float)},
    {"task_id": "grid_04", "cube_spawn_pos": np.array([0.506, 0.030, 1.050], dtype=float)},
    {"task_id": "grid_05", "cube_spawn_pos": np.array([0.516, 0.030, 1.050], dtype=float)},
    {"task_id": "grid_06", "cube_spawn_pos": np.array([0.526, 0.030, 1.050], dtype=float)},
    {"task_id": "grid_07", "cube_spawn_pos": np.array([0.506, 0.050, 1.050], dtype=float)},
    {"task_id": "grid_08", "cube_spawn_pos": np.array([0.516, 0.050, 1.050], dtype=float)},
    {"task_id": "grid_09", "cube_spawn_pos": np.array([0.526, 0.050, 1.050], dtype=float)},
]


def _set_free_body_pos(model, data, body_name: str, pos: np.ndarray) -> None:
    core.set_free_body_pos(model, data, body_name, np.asarray(pos, dtype=float))


def _try_set_ctrl(model, data, name: str, value: float) -> None:
    try:
        core.set_ctrl(model, data, name, value)
    except Exception:
        pass


def prepare_scene_for_detection(model, data, cube_spawn_pos: np.ndarray, viewer=None, realtime: bool = False) -> np.ndarray:
    """Place the cube, let it settle, and return the stable body position."""
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
    print("[VISION] raw_world_pos:", det.raw_world_pos)
    print("[VISION] calibration_offset:", det.calibration_offset)
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


def run_vision_calibration(model, data, detector: OrangeCubeDetector, viewer=None, realtime: bool = False, calibration_path: Path = CALIBRATION_PATH) -> np.ndarray:
    print("\n" + "=" * 100)
    print("V18.3 VISION CALIBRATION")
    print("说明：用多个已知仿真位置计算 camera backprojection 的 XY 偏差，并保存到 JSON。")
    print("=" * 100)

    rows: list[dict] = []
    offsets: list[np.ndarray] = []
    CALIBRATION_SAMPLES_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_SAMPLES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task_id",
            "spawn_x", "spawn_y", "spawn_z",
            "true_x", "true_y", "true_z",
            "raw_x", "raw_y", "raw_z",
            "offset_x", "offset_y",
            "raw_error_xy",
        ])

        for case in VISION_CALIBRATION_CASES:
            task_id = str(case["task_id"])
            cube_spawn = np.asarray(case["cube_spawn_pos"], dtype=float)
            stable = prepare_scene_for_detection(model, data, cube_spawn, viewer=viewer, realtime=realtime)
            det = detector.detect(
                task_id=task_id,
                planner_spawn_z=float(cube_spawn[2]),
                stable_cube_z=float(stable[2]),
                apply_calibration=False,
            )
            raw_xy = det.raw_world_pos[:2]
            true_xy = stable[:2]
            offset_xy = true_xy - raw_xy
            raw_err = float(np.linalg.norm(raw_xy - true_xy))
            offsets.append(np.array([offset_xy[0], offset_xy[1], 0.0], dtype=float))
            row = {
                "task_id": task_id,
                "spawn": [float(x) for x in cube_spawn.tolist()],
                "true_stable": [float(x) for x in stable.tolist()],
                "raw_world": [float(x) for x in det.raw_world_pos.tolist()],
                "offset_xy": [float(offset_xy[0]), float(offset_xy[1])],
                "raw_error_xy": raw_err,
                "pixel_center": [float(det.pixel_center[0]), float(det.pixel_center[1])],
            }
            rows.append(row)
            writer.writerow([
                task_id,
                float(cube_spawn[0]), float(cube_spawn[1]), float(cube_spawn[2]),
                float(stable[0]), float(stable[1]), float(stable[2]),
                float(det.raw_world_pos[0]), float(det.raw_world_pos[1]), float(det.raw_world_pos[2]),
                float(offset_xy[0]), float(offset_xy[1]),
                raw_err,
            ])
            print(f"[CALIB] {task_id:14s} true=({stable[0]:.4f},{stable[1]:.4f}) raw=({det.raw_world_pos[0]:.4f},{det.raw_world_pos[1]:.4f}) offset=({offset_xy[0]:+.4f},{offset_xy[1]:+.4f}) raw_err={raw_err:.4f}")

    if not offsets:
        raise RuntimeError("calibration failed: no offsets collected")
    offset = np.median(np.vstack(offsets), axis=0)
    detector.set_calibration_offset(offset)
    save_calibration_offset(calibration_path, offset, rows, camera_name=CAMERA_NAME)
    print("-" * 100)
    print(f"[CALIB] final median xy_offset = ({offset[0]:+.5f}, {offset[1]:+.5f}, 0.00000)")
    print("[CALIB] saved:", calibration_path)
    print("[CALIB] samples csv:", CALIBRATION_SAMPLES_CSV)
    return offset


def run_vision_grid_test(model, data, detector: OrangeCubeDetector, viewer=None, realtime: bool = False) -> None:
    print("\n" + "=" * 100)
    print("V18.3 VISION GRID TEST")
    print("说明：只测试相机识别精度，不执行抓取。")
    print("=" * 100)

    GRID_TEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    errors: list[float] = []
    with open(GRID_TEST_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task_id",
            "spawn_x", "spawn_y", "spawn_z",
            "true_x", "true_y", "true_z",
            "raw_x", "raw_y", "raw_z",
            "calib_dx", "calib_dy",
            "estimated_x", "estimated_y", "estimated_z",
            "error_xy",
            "debug_image_path",
        ])
        for case in VISION_GRID_TEST_CASES:
            task_id = str(case["task_id"])
            cube_spawn = np.asarray(case["cube_spawn_pos"], dtype=float)
            stable = prepare_scene_for_detection(model, data, cube_spawn, viewer=viewer, realtime=realtime)
            det = detector.detect(task_id=task_id, planner_spawn_z=float(cube_spawn[2]), stable_cube_z=float(stable[2]))
            err = float(np.linalg.norm(det.stable_world_pos[:2] - stable[:2]))
            errors.append(err)
            writer.writerow([
                task_id,
                float(cube_spawn[0]), float(cube_spawn[1]), float(cube_spawn[2]),
                float(stable[0]), float(stable[1]), float(stable[2]),
                float(det.raw_world_pos[0]), float(det.raw_world_pos[1]), float(det.raw_world_pos[2]),
                float(det.calibration_offset[0]), float(det.calibration_offset[1]),
                float(det.stable_world_pos[0]), float(det.stable_world_pos[1]), float(det.stable_world_pos[2]),
                err,
                det.debug_image_path,
            ])
            print(f"[GRID] {task_id:8s} true=({stable[0]:.4f},{stable[1]:.4f}) est=({det.stable_world_pos[0]:.4f},{det.stable_world_pos[1]:.4f}) err_xy={err:.4f}")

    if errors:
        arr = np.asarray(errors, dtype=float)
        print("-" * 100)
        print(f"[GRID] mean_err={arr.mean():.4f} m, max_err={arr.max():.4f} m, n={len(arr)}")
    print("[GRID] saved:", GRID_TEST_CSV)


def write_vision_report(tasks: List[core.PickPlaceTask], results: List[core.TaskResult], calibration_source: str, calibration_offset: np.ndarray) -> None:
    core.ensure_output_dir()
    success_count = sum(1 for r in results if r.place_success)
    lines: List[str] = []
    lines.append("# OpenArm V18.3 Vision Grasp Demo")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("This demo adds MuJoCo camera perception before task planning. The system renders an RGB image from an overhead camera, detects the orange cube by color segmentation, estimates its world position, and feeds the detected pose into the bimanual task planner.")
    lines.append("")
    lines.append("## Pipeline")
    lines.append("")
    lines.append("```text")
    lines.append("Camera RGB render -> orange cube segmentation -> pixel center -> world pose estimate -> XY calibration -> select arm -> path planning -> collision check -> pick-and-place")
    lines.append("```")
    lines.append("")
    lines.append("## Vision Calibration")
    lines.append("")
    lines.append(f"- Calibration source: `{calibration_source}`")
    lines.append(f"- XY offset: `({calibration_offset[0]:+.5f}, {calibration_offset[1]:+.5f}, {calibration_offset[2]:+.5f})`")
    lines.append(f"- Calibration file: `{CALIBRATION_PATH}`")
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
    lines.append(f"- Grid test CSV: `{GRID_TEST_CSV}`")
    VISION_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("[V18.3] vision report:", VISION_REPORT_PATH)


def print_v18_header(cases: List[Dict[str, Any]], right_adapter: str, calibration_source: str, calibration_offset: np.ndarray) -> None:
    print("=" * 100)
    print("OpenArm V18.3 Vision + Task Planner Demo")
    print("=" * 100)
    print("新增功能：MuJoCo 相机视觉识别 orange_cube；可视化相机 marker；支持自动标定和网格精度测试。")
    print("camera:", CAMERA_NAME)
    print("XML_PATH:", core.XML_PATH)
    print("VISION_XML_PATH:", VISION_XML_PATH)
    print("right_adapter:", right_adapter)
    print("vision_calibration_source:", calibration_source)
    print("vision_calibration_offset:", calibration_offset)
    print("cases:")
    for c in cases:
        p = c["cube_spawn_pos"]
        print(f"  {c['task_id']:18s} spawn=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) | {c.get('description', '')}")
    print("=" * 100)


def resolve_calibration(args) -> tuple[np.ndarray, str, Path]:
    calibration_path = Path(args.calibration_file) if args.calibration_file else CALIBRATION_PATH
    if args.no_load_calibration:
        return DEFAULT_VISION_XY_CALIBRATION_OFFSET.copy(), "default_forced_no_load", calibration_path
    offset, source = load_calibration_offset(calibration_path, DEFAULT_VISION_XY_CALIBRATION_OFFSET)
    return offset, source, calibration_path


def run(args) -> None:
    patch_v18_output_paths()
    ensure_vision_xml(core.XML_PATH, VISION_XML_PATH, camera_name=CAMERA_NAME)
    core.XML_PATH = VISION_XML_PATH
    core.ensure_output_dir()
    init_detection_log(DETECTION_LOG_PATH)

    calibration_offset, calibration_source, calibration_path = resolve_calibration(args)
    cases = VISION_BOTH_ARM_CASES if args.both_arms else VISION_RIGHT_CASES
    print_v18_header(cases, right_adapter=args.right_adapter, calibration_source=calibration_source, calibration_offset=calibration_offset)

    model = mujoco.MjModel.from_xml_path(str(core.XML_PATH))
    data = mujoco.MjData(model)
    detector = OrangeCubeDetector(model, data, debug_dir=VISION_DEBUG_DIR, calibration_offset=calibration_offset)

    def maybe_calibrate(viewer=None, realtime=False) -> tuple[np.ndarray, str]:
        nonlocal calibration_offset, calibration_source
        if args.calibrate_vision:
            calibration_offset = run_vision_calibration(model, data, detector, viewer=viewer, realtime=realtime, calibration_path=calibration_path)
            calibration_source = f"generated:{calibration_path}"
        return calibration_offset, calibration_source

    def build_tasks(viewer=None, realtime=False) -> List[core.PickPlaceTask]:
        tasks: List[core.PickPlaceTask] = []
        for case in cases:
            task = detect_case_to_task(model, data, detector, case, viewer=viewer, realtime=realtime)
            tasks.append(task)
        core.write_execution_plan(tasks)
        return tasks

    if args.no_viewer:
        maybe_calibrate(viewer=None, realtime=False)
        if args.calibrate_only:
            return
        if args.vision_grid_test:
            run_vision_grid_test(model, data, detector, viewer=None, realtime=False)
            return
        tasks = build_tasks(viewer=None, realtime=False)
        planner = core.BimanualTaskPlanner(
            model,
            data,
            viewer=None,
            realtime=False,
            max_retries=args.max_retries,
            enable_fallback=(args.allow_cross_arm_fallback and not args.no_fallback),
            simulate_recovery=args.simulate_recovery,
            right_adapter_mode=args.right_adapter,
        )
        results = planner.run(tasks)
        write_vision_report(tasks, results, calibration_source=calibration_source, calibration_offset=calibration_offset)
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        print("viewer 已打开。2 秒后开始视觉识别 + 抓取 demo。")
        time.sleep(2.0)
        maybe_calibrate(viewer=viewer, realtime=True)
        if args.calibrate_only:
            print("[V18.3] calibrate-only 完成。")
            if args.hold_viewer:
                print("viewer 保持运行。关闭 viewer 或 Ctrl+C 结束。")
                while viewer.is_running():
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    time.sleep(model.opt.timestep)
            return
        if args.vision_grid_test:
            run_vision_grid_test(model, data, detector, viewer=viewer, realtime=True)
            if args.hold_viewer:
                print("viewer 保持运行。关闭 viewer 或 Ctrl+C 结束。")
                while viewer.is_running():
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    time.sleep(model.opt.timestep)
            return

        tasks = build_tasks(viewer=viewer, realtime=True)
        planner = core.BimanualTaskPlanner(
            model,
            data,
            viewer=viewer,
            realtime=True,
            max_retries=args.max_retries,
            enable_fallback=(args.allow_cross_arm_fallback and not args.no_fallback),
            simulate_recovery=args.simulate_recovery,
            right_adapter_mode=args.right_adapter,
        )
        results = planner.run(tasks)
        write_vision_report(tasks, results, calibration_source=calibration_source, calibration_offset=calibration_offset)

        if args.hold_viewer:
            print("viewer 保持运行。关闭 viewer 或 Ctrl+C 结束。")
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)


def parse_args():
    parser = argparse.ArgumentParser(description="OpenArm V18.3 vision perception + grasp demo")
    parser.add_argument("--both-arms", action="store_true", help="run left vision task and right vision task")
    parser.add_argument("--right-adapter", choices=["rule", "rl", "rl_fallback"], default="rule", help="right arm executor")
    parser.add_argument("--simulate-recovery", action="store_true", help="inject one failure before real action")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--allow-cross-arm-fallback", action="store_true", help="allow planner to switch to the other arm after retries; default off for vision demo")
    parser.add_argument("--no-fallback", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--calibrate-vision", action="store_true", help="generate/update vision calibration JSON before normal run")
    parser.add_argument("--calibrate-only", action="store_true", help="only run calibration and exit")
    parser.add_argument("--vision-grid-test", action="store_true", help="only test vision accuracy on a 3x3 grid; no grasp")
    parser.add_argument("--calibration-file", default="", help="optional calibration JSON path")
    parser.add_argument("--no-load-calibration", action="store_true", help="ignore saved calibration and use default V18.2 offset")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--no-hold-viewer", dest="hold_viewer", action="store_false")
    parser.set_defaults(hold_viewer=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()

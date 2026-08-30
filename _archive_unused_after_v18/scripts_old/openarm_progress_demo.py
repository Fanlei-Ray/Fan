from pathlib import Path
import argparse
import csv
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# OpenArm progress demo for presentation
#
# 推荐汇报现场运行：
#   python scripts\openarm_progress_demo.py
#
# 这个文件的目标是“稳定展示进度”，不是继续调参。
# 默认只运行已经验证成功的右臂 rule expert pick-and-place。
# 左/右 selector 的逻辑会在终端打印，左臂分支暂不强制执行，避免现场失败。
#
# 文件放置位置：
#   openarm_mujoco-master\scripts\openarm_progress_demo.py
# ============================================================


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import right_rule_pick_place as right_rule
except Exception as exc:
    raise RuntimeError(
        "导入 scripts/right_rule_pick_place.py 失败。请确认本文件放在项目的 scripts 目录下。"
    ) from exc


XML_PATH = ROOT / "v2" / "demo.xml"
OUTPUT_DIR = ROOT / "outputs" / "presentation_demo"
LOG_PATH = OUTPUT_DIR / "openarm_progress_demo_log.csv"


# ============================================================
# Demo cases
#
# right_fixed_success 是目前最稳的右臂展示点。
# 右臂 rule expert 已经在这个点上验证：pick_success=True, place_success=True。
# ============================================================

RIGHT_SHOWCASE_CASES = [
    {
        "name": "right_fixed_success",
        "cube_pos": np.array([0.516, 0.050, 1.050], dtype=float),
        "description": "右臂固定点 pick-and-place 成功展示",
    },
]

# selector 原型：y >= 0.02 走右臂，否则走左臂。
# 当前稳定演示只执行右臂；左臂分支保留为进度说明。
SELECT_RIGHT_IF_CUBE_Y_GE = 0.020

SELECTOR_PREVIEW_CASES = [
    {
        "name": "left_workspace_preview",
        "cube_pos": np.array([0.516, -0.035, 1.050], dtype=float),
        "note": "selector 会选择 left；左臂 TCP/offset 仍在调试，不作为现场稳定动作演示",
    },
    {
        "name": "right_workspace_execute",
        "cube_pos": np.array([0.516, 0.050, 1.050], dtype=float),
        "note": "selector 会选择 right；当前现场稳定执行这一支",
    },
]


# ============================================================
# Right arm best config
#
# 来自右臂 fixed pick/place 成功结果：
#   config_name: tcp_x-0.006_y-0.020_z+0.055__j7_m060
#   pick_success=True, place_success=True
# ============================================================

RIGHT_BEST_CONFIG = {
    "name": "right_best_tcp_x-0.006_y-0.020_z+0.055__j7_m060",
    "site_type": "tcp",

    "pregrasp_offset": np.array([-0.006, -0.020, 0.140], dtype=float),
    "grasp_offset": np.array([-0.006, -0.020, 0.055], dtype=float),

    "preplace_offset": np.array([0.0, 0.0, 0.140], dtype=float),
    "place_offset": np.array([0.0, 0.0, 0.080], dtype=float),

    "joint_biases": {
        "right_joint7_ctrl": -0.060,
    },
}


def select_arm(cube_pos: np.ndarray) -> str:
    """Simple bimanual selector prototype."""
    if float(cube_pos[1]) >= SELECT_RIGHT_IF_CUBE_Y_GE:
        return "right"
    return "left"


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def init_log() -> None:
    ensure_output_dir()
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case",
                "arm",
                "cube_x",
                "cube_y",
                "cube_z",
                "pick_success",
                "place_success",
                "lift_delta",
                "final_lift_delta",
                "xy_dist",
                "z_margin",
                "cube_final_x",
                "cube_final_y",
                "cube_final_z",
                "frame_final_x",
                "frame_final_y",
                "frame_final_z",
            ]
        )


def append_log(case_name: str, arm: str, cube_pos: np.ndarray, result: dict) -> None:
    ensure_output_dir()
    cube_final = np.asarray(result.get("cube_final", [np.nan, np.nan, np.nan]), dtype=float)
    frame_final = np.asarray(result.get("frame_final", [np.nan, np.nan, np.nan]), dtype=float)

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                case_name,
                arm,
                float(cube_pos[0]),
                float(cube_pos[1]),
                float(cube_pos[2]),
                bool(result.get("pick_success", False)),
                bool(result.get("place_success", False)),
                float(result.get("lift_delta", 0.0)),
                float(result.get("final_lift_delta", 0.0)),
                float(result.get("xy_dist", 0.0)),
                float(result.get("z_margin", 0.0)),
                float(cube_final[0]),
                float(cube_final[1]),
                float(cube_final[2]),
                float(frame_final[0]),
                float(frame_final[1]),
                float(frame_final[2]),
            ]
        )


def print_presentation_plan() -> None:
    print("=" * 90)
    print("OpenArm 汇报展示脚本")
    print("=" * 90)
    print("展示目标：")
    print("  1. MuJoCo 中打开 OpenArm 双臂工作台场景。")
    print("  2. 展示右臂 rule expert 自动完成 orange_cube -> black_frame 的 pick-and-place。")
    print("  3. 终端打印 bimanual selector 原型：cube_y >= 0.02 选右臂，否则选左臂。")
    print("  4. 当前稳定现场 demo 只执行右臂成功分支；左臂分支作为下一步继续调 TCP/offset。")
    print("")
    print("为什么现场先展示右臂稳定分支：")
    print("  - 右臂 fixed pick/place 已验证成功。")
    print("  - 右臂随机采集 156 次尝试得到 120 条成功样本，说明 rule expert 已可用。")
    print("  - 左臂历史 BC 已有进展，但当前 selector rule 版本的 left_gripper_tcp offset 还需要继续校准。")
    print("")
    print("selector 原型：")
    print(f"  if cube_y >= {SELECT_RIGHT_IF_CUBE_Y_GE:.3f}: use right arm")
    print(f"  else: use left arm")
    print("=" * 90)

    for case in SELECTOR_PREVIEW_CASES:
        cube_pos = case["cube_pos"]
        arm = select_arm(cube_pos)
        print(
            f"selector_preview: {case['name']:24s} "
            f"cube=({cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}) "
            f"-> selected_arm={arm:5s} | {case['note']}"
        )
    print("=" * 90)


def run_right_case(model, data, viewer, case: dict, realtime: bool) -> dict:
    cube_pos = np.asarray(case["cube_pos"], dtype=float)

    print("")
    print("#" * 90)
    print(f"RIGHT ARM DEMO CASE: {case['name']}")
    print("#" * 90)
    print("description:", case.get("description", ""))
    print("cube_pos:", cube_pos)
    print("selector selected arm:", select_arm(cube_pos))
    print("right config:", RIGHT_BEST_CONFIG["name"])
    print("pregrasp_offset:", RIGHT_BEST_CONFIG["pregrasp_offset"])
    print("grasp_offset:", RIGHT_BEST_CONFIG["grasp_offset"])
    print("joint_biases:", RIGHT_BEST_CONFIG["joint_biases"])
    print("#" * 90)

    if select_arm(cube_pos) != "right":
        raise RuntimeError("这个展示 case 没有被 selector 选为 right，请检查 cube_y。")

    right_rule.FIXED_CUBE_POS = cube_pos.copy()
    site_name = right_rule.choose_right_site(model)

    print("使用右臂 site:", site_name)
    print("开始执行右臂 pick-and-place...")

    result = right_rule.run_trial(
        model=model,
        site_name=site_name,
        config=RIGHT_BEST_CONFIG,
        do_place=True,
        viewer=viewer,
        realtime=realtime,
        data=data,
    )

    out = {
        "arm": "right",
        "pick_success": bool(result["pick_success"]),
        "place_success": bool(result["place_success"]),
        "lift_delta": float(result["lift_delta"]),
        "final_lift_delta": float(result["final_lift_delta"]),
        "xy_dist": float(result["xy_dist"]),
        "z_margin": float(result["z_margin"]),
        "cube_final": np.asarray(result["cube_final"], dtype=float).copy(),
        "frame_final": np.asarray(result["frame_final"], dtype=float).copy(),
    }

    print("")
    print("=" * 90)
    print(f"CASE RESULT: {case['name']}")
    print("=" * 90)
    print("arm:", out["arm"])
    print("pick_success:", out["pick_success"])
    print("place_success:", out["place_success"])
    print("lift_delta:", out["lift_delta"])
    print("final_lift_delta:", out["final_lift_delta"])
    print("xy_dist:", out["xy_dist"])
    print("z_margin:", out["z_margin"])
    print("cube_final:", out["cube_final"])
    print("frame_final:", out["frame_final"])
    print("=" * 90)

    append_log(case["name"], "right", cube_pos, out)
    return out


def run_with_viewer(args) -> None:
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()

        print("viewer 已打开。")
        print("2 秒后开始展示。")
        time.sleep(2.0)

        results = []
        for i, case in enumerate(RIGHT_SHOWCASE_CASES[: args.cases], start=1):
            print("")
            print(f"展示进度：{i}/{min(args.cases, len(RIGHT_SHOWCASE_CASES))}")
            result = run_right_case(
                model=model,
                data=data,
                viewer=viewer,
                case=case,
                realtime=True,
            )
            results.append((case, result))

            if i < min(args.cases, len(RIGHT_SHOWCASE_CASES)):
                print("暂停 2 秒进入下一个 case。")
                pause_steps = int(2.0 / model.opt.timestep)
                for _ in range(pause_steps):
                    if not viewer.is_running():
                        break
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    time.sleep(model.opt.timestep)

        print_summary(results)

        if args.hold_viewer:
            print("")
            print("viewer 保持运行。关闭 viewer 或 Ctrl+C 结束。")
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)


def run_headless(args) -> None:
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    results = []
    for case in RIGHT_SHOWCASE_CASES[: args.cases]:
        result = run_right_case(
            model=model,
            data=data,
            viewer=None,
            case=case,
            realtime=False,
        )
        results.append((case, result))

    print_summary(results)


def print_summary(results) -> None:
    print("")
    print("=" * 90)
    print("OpenArm progress demo 总结")
    print("=" * 90)

    success_count = 0

    for case, result in results:
        if result["place_success"]:
            success_count += 1

        cube_pos = case["cube_pos"]
        print(
            f"{case['name']:24s} "
            f"cube=({cube_pos[0]:.3f},{cube_pos[1]:.3f},{cube_pos[2]:.3f}) "
            f"arm={result['arm']:5s} "
            f"pick={result['pick_success']} "
            f"place={result['place_success']} "
            f"lift={result['lift_delta']:.4f} "
            f"xy={result['xy_dist']:.4f} "
            f"z_margin={result['z_margin']:.4f}"
        )

    print("")
    print(f"total place_success: {success_count}/{len(results)}")
    print("LOG_PATH:", LOG_PATH)
    print("=" * 90)

    if success_count == len(results) and len(results) > 0:
        print("展示结论：右臂 rule expert 稳定完成 pick-and-place；双臂 selector 框架已搭建。")
    else:
        print("提示：本次展示 case 未全部成功。汇报现场建议只保留 right_fixed_success。")


def parse_args():
    parser = argparse.ArgumentParser(
        description="OpenArm progress presentation demo. Default: stable right-arm pick-and-place viewer demo."
    )

    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="不打开 viewer，快速 headless 测试。",
    )

    parser.add_argument(
        "--cases",
        type=int,
        default=1,
        help="运行多少个右臂展示 case。默认 1。",
    )

    parser.add_argument(
        "--no-hold-viewer",
        dest="hold_viewer",
        action="store_false",
        help="执行结束后不保持 viewer。",
    )

    parser.set_defaults(hold_viewer=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not XML_PATH.exists():
        raise FileNotFoundError(f"找不到 XML：{XML_PATH}")

    ensure_output_dir()
    init_log()

    print_presentation_plan()

    if args.no_viewer:
        run_headless(args)
    else:
        run_with_viewer(args)


if __name__ == "__main__":
    main()

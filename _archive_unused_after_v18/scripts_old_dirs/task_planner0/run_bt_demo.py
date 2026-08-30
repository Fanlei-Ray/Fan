from __future__ import annotations

"""OpenArm V12: behavior-tree execution demo.

Run from project root:
    python scripts\task_planner\run_bt_demo.py
    python scripts\task_planner\run_bt_demo.py --queue-demo
    python scripts\task_planner\run_bt_demo.py --simulate-recovery
    python scripts\task_planner\run_bt_demo.py --no-viewer

Compared with V11, this file actually executes every task through a tiny
behavior tree:

Sequence(PickPlaceTask)
  PerceiveObject
  SelectArm
  SafetyCheck
  PlanPath
  PathSafe?
  ParkInactiveArm
  PickAndPlaceWithRecovery
  VerifySuccess

The underlying arm execution still uses the stable V10/V11 adapters, so we do
not break the already working demo.
"""

from pathlib import Path
import argparse
import csv
import sys
import time
from typing import List

import mujoco
import mujoco.viewer

TASK_PLANNER_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = TASK_PLANNER_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import task_planner.core as core
from task_planner.behavior_tree import ActionNode, BTContext, BTStatus, ConditionNode, SequenceNode


BT_TRACE_PATH = core.OUTPUT_DIR / "behavior_tree_trace_v12.csv"


def patch_v12_output_paths() -> None:
    core.LOG_PATH = core.OUTPUT_DIR / "task_log_v12_bt.csv"
    core.PLAN_PATH = core.OUTPUT_DIR / "execution_plan_v12_bt.csv"
    core.SUMMARY_PATH = core.OUTPUT_DIR / "task_summary_v12_bt.json"
    core.REPORT_PATH = core.OUTPUT_DIR / "presentation_report_v12_bt.md"
    core.STATE_MACHINE_PATH = core.OUTPUT_DIR / "state_machine_v12_bt.mmd"
    core.RUNBOOK_PATH = core.OUTPUT_DIR / "demo_runbook_v12_bt.txt"
    core.PATH_PLAN_PATH = core.OUTPUT_DIR / "path_plan_v12_bt.csv"
    core.COLLISION_LOG_PATH = core.OUTPUT_DIR / "collision_log_v12_bt.csv"


def init_bt_trace() -> None:
    core.ensure_output_dir()
    with open(BT_TRACE_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "task_id", "node_name", "node_type", "status", "message"])


def append_bt_trace(ctx: BTContext) -> None:
    core.ensure_output_dir()
    with open(BT_TRACE_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in ctx.trace:
            writer.writerow([row.timestamp, row.task_id, row.node_name, row.node_type, row.status, row.message])


def build_pick_place_tree() -> SequenceNode:
    def init_task(ctx: BTContext) -> bool:
        task = ctx.task
        print("\n" + "=" * 100)
        print(f"BT TASK START: {task.task_id}")
        print("description:", task.description)
        print("=" * 100)
        ctx.planner.logger.log(task, core.PlannerState.INIT, "", message="bt task init")
        return True

    def perceive(ctx: BTContext) -> bool:
        ctx.obs = ctx.planner.perceive(ctx.task)
        ctx.planner.logger.log(ctx.task, core.PlannerState.PERCEIVE, "", message="bt object pose acquired")
        return "cube_pos" in ctx.obs

    def select_arm(ctx: BTContext) -> bool:
        ctx.selected_arm = ctx.planner.select_arm(ctx.obs["cube_pos"])
        ctx.planner.logger.log(ctx.task, core.PlannerState.SELECT_ARM, ctx.selected_arm, message="bt arm selected")
        return ctx.selected_arm in ["left", "right"]

    def safety_check(ctx: BTContext) -> bool:
        ctx.safety_snapshot = ctx.planner.safety.safety_check(ctx.selected_arm, ctx.obs["cube_pos"])
        collision_summary = ctx.planner.collision.snapshot(
            ctx.task.task_id,
            core.PlannerState.SAFETY_CHECK.name,
            ctx.selected_arm,
            message="bt pre-action collision snapshot",
        )
        ctx.planner.logger.log(
            ctx.task,
            core.PlannerState.SAFETY_CHECK,
            ctx.selected_arm,
            safety=ctx.safety_snapshot,
            message=f"bt safety pre-check | collision_dangerous={collision_summary['dangerous_count']}",
        )
        return bool(ctx.safety_snapshot.workspace_ok) and int(collision_summary["dangerous_count"]) == 0

    def plan_path(ctx: BTContext) -> bool:
        ctx.waypoints = ctx.planner.path_planner.plan_pick_place_path(ctx.task, ctx.selected_arm, ctx.obs["cube_pos"])
        ctx.path_safe = ctx.planner.path_planner.path_is_safe(ctx.waypoints)
        ctx.planner.path_planner.print_path(ctx.task, ctx.selected_arm, ctx.waypoints)
        ctx.planner.path_planner.write_path(ctx.waypoints)
        collision_summary = ctx.planner.collision.snapshot(
            ctx.task.task_id,
            core.PlannerState.PLAN_PATH.name,
            ctx.selected_arm,
            message="bt path planning collision snapshot",
        )
        ctx.planner.logger.log(
            ctx.task,
            core.PlannerState.PLAN_PATH,
            ctx.selected_arm,
            safety=ctx.safety_snapshot,
            message=f"bt path planned, waypoints={len(ctx.waypoints)}, path_safe={ctx.path_safe}, collision_dangerous={collision_summary['dangerous_count']}",
        )
        return int(collision_summary["dangerous_count"]) == 0

    def park_inactive(ctx: BTContext) -> bool:
        ctx.safety_snapshot = ctx.planner.safety.park_inactive_arm(ctx.selected_arm, ctx.obs["cube_pos"])
        collision_summary = ctx.planner.collision.snapshot(
            ctx.task.task_id,
            core.PlannerState.PARK_INACTIVE_ARM.name,
            ctx.selected_arm,
            message="bt after inactive arm park",
        )
        ctx.planner.logger.log(
            ctx.task,
            core.PlannerState.PARK_INACTIVE_ARM,
            ctx.selected_arm,
            safety=ctx.safety_snapshot,
            message=f"bt inactive arm parked / safe | collision_dangerous={collision_summary['dangerous_count']}",
        )
        return int(collision_summary["dangerous_count"]) == 0

    def execute_pick_place(ctx: BTContext) -> bool:
        ctx.result = ctx.planner.execute_with_recovery(
            task=ctx.task,
            selected_arm=ctx.selected_arm,
            cube_pos=ctx.obs["cube_pos"],
            safety_snapshot=ctx.safety_snapshot,
        )
        ctx.selected_arm = ctx.result.selected_arm
        ctx.planner.collision.snapshot(
            ctx.task.task_id,
            core.PlannerState.EXECUTE_PICK_PLACE.name,
            ctx.selected_arm,
            message="bt post-action collision snapshot",
        )
        ctx.planner.safety.release_after_task()
        return bool(ctx.result.pick_success)

    def verify_success(ctx: BTContext) -> bool:
        ctx.planner.print_result(ctx.result)
        ok = bool(ctx.result.place_success)
        ctx.planner.logger.log(
            ctx.task,
            core.PlannerState.VERIFY,
            ctx.selected_arm,
            result=ctx.result,
            message="bt success" if ok else "bt failed",
        )
        ctx.planner.logger.log(
            ctx.task,
            core.PlannerState.DONE if ok else core.PlannerState.FAILED,
            ctx.selected_arm,
            result=ctx.result,
            message="bt task done" if ok else "bt task failed",
        )
        print("=" * 100)
        print(f"BT TASK END: {ctx.task.task_id} -> {'DONE' if ok else 'FAILED'}")
        print("=" * 100)
        return ok

    return SequenceNode(
        "PickPlaceTaskSequence",
        [
            ActionNode("InitTask", init_task),
            ActionNode("PerceiveObject", perceive, lambda ctx: f"cube_pos={ctx.obs['cube_pos']}"),
            ActionNode("SelectArm", select_arm, lambda ctx: f"selected_arm={ctx.selected_arm}"),
            ActionNode("SafetyCheck", safety_check, lambda ctx: f"workspace_ok={ctx.safety_snapshot.workspace_ok}"),
            ActionNode("PlanPath", plan_path, lambda ctx: f"waypoints={len(ctx.waypoints)}"),
            ConditionNode("PathSafe", lambda ctx: ctx.path_safe, lambda ctx: f"path_safe={ctx.path_safe}"),
            ActionNode("ParkInactiveArm", park_inactive, lambda ctx: f"inactive_arm={ctx.safety_snapshot.inactive_arm}"),
            ActionNode("PickAndPlaceWithRecovery", execute_pick_place, lambda ctx: f"pick={ctx.result.pick_success}, place={ctx.result.place_success}"),
            ConditionNode("VerifySuccess", verify_success, lambda ctx: f"place_success={ctx.result.place_success}"),
        ],
    )


class BehaviorTreeTaskRunner:
    def __init__(self, planner: core.BimanualTaskPlanner):
        self.planner = planner
        self.tree = build_pick_place_tree()

    def run(self, tasks: List[core.PickPlaceTask]) -> List[core.TaskResult]:
        core.print_presentation_header(tasks)
        print("\n" + "#" * 100)
        print("V12 Behavior Tree Executor")
        print("#" * 100)
        print("真正执行结构：")
        print("  Sequence(PickPlaceTask)")
        print("    -> InitTask")
        print("    -> PerceiveObject")
        print("    -> SelectArm")
        print("    -> SafetyCheck")
        print("    -> PlanPath")
        print("    -> PathSafe")
        print("    -> ParkInactiveArm")
        print("    -> PickAndPlaceWithRecovery")
        print("    -> VerifySuccess")
        print("#" * 100)

        results: List[core.TaskResult] = []
        for i, task in enumerate(tasks, start=1):
            print(f"\n[BT RUNNER] Running task {i}/{len(tasks)}: {task.task_id}")
            ctx = BTContext(task=task, planner=self.planner)
            status = self.tree.tick(ctx)
            append_bt_trace(ctx)
            if ctx.result is not None:
                results.append(ctx.result)
            else:
                result = self.planner.make_failure_result(task, ctx.selected_arm or "", message=f"BT failed before action: {status.name}")
                results.append(result)

            if i < len(tasks):
                print("[BT RUNNER] pause 1.5s before next task")
                core.sim_steps(
                    self.planner.model,
                    self.planner.data,
                    steps=int(1.5 / self.planner.model.opt.timestep),
                    viewer=self.planner.viewer,
                    realtime=self.planner.realtime,
                )

        core.print_final_summary(results)
        core.write_summary_json(results)
        core.write_presentation_assets(tasks, results)
        print("BT_TRACE_PATH:", BT_TRACE_PATH)
        return results


def parse_args():
    parser = argparse.ArgumentParser(description="OpenArm V12 behavior-tree task planner demo")
    parser.add_argument("--no-viewer", action="store_true", help="不打开 viewer，快速 headless 测试")
    parser.add_argument("--right-only", action="store_true", help="只运行右臂稳定展示任务")
    parser.add_argument("--queue-demo", action="store_true", help="运行 4 个任务的队列演示：left/right/left/right")
    parser.add_argument("--simulate-recovery", action="store_true", help="在第一个任务第一次尝试时注入一次模拟失败，展示 retry 恢复逻辑")
    parser.add_argument("--max-retries", type=int, default=1, help="失败后同一只手最多重试次数，默认 1")
    parser.add_argument("--no-fallback", dest="enable_fallback", action="store_false", help="失败后不切换到另一只手兜底")
    parser.add_argument("--no-hold-viewer", dest="hold_viewer", action="store_false", help="结束后不保持 viewer")
    parser.set_defaults(hold_viewer=True, enable_fallback=True)
    return parser.parse_args()


def select_tasks(args) -> List[core.PickPlaceTask]:
    if args.right_only:
        raw = core.RIGHT_ONLY_TASKS
    elif args.queue_demo:
        raw = core.QUEUE_TASKS
    else:
        raw = core.DEFAULT_TASKS
    return core.make_tasks(raw)


def run_with_model(args, model, data, viewer, realtime: bool) -> None:
    tasks = select_tasks(args)
    core.write_execution_plan(tasks)
    init_bt_trace()

    planner = core.BimanualTaskPlanner(
        model,
        data,
        viewer=viewer,
        realtime=realtime,
        max_retries=args.max_retries,
        enable_fallback=args.enable_fallback,
        simulate_recovery=args.simulate_recovery,
    )
    runner = BehaviorTreeTaskRunner(planner)
    runner.run(tasks)


def main() -> None:
    args = parse_args()
    patch_v12_output_paths()
    core.ensure_output_dir()

    if not core.XML_PATH.exists():
        raise FileNotFoundError(f"找不到 XML：{core.XML_PATH}")

    model = mujoco.MjModel.from_xml_path(str(core.XML_PATH))
    data = mujoco.MjData(model)

    if args.no_viewer:
        run_with_model(args, model, data, None, realtime=False)
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        print("viewer 已打开。2 秒后开始 V12 行为树任务规划 demo。")
        time.sleep(2.0)
        run_with_model(args, model, data, viewer, realtime=True)

        if args.hold_viewer:
            print("viewer 保持运行。关闭 viewer 或 Ctrl+C 结束。")
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()

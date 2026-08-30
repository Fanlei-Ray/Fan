from __future__ import annotations

"""OpenArm V17: hybrid FSM + Behavior Tree + right-arm RL adapter with rule fallback.

Run from project root:
    python scripts\task_planner\run_hybrid_demo.py
    python scripts\task_planner\run_hybrid_demo.py --queue-demo
    python scripts\task_planner\run_hybrid_demo.py --simulate-recovery
    python scripts\task_planner\run_hybrid_demo.py --no-viewer

V13 combines the two execution styles:

1. Behavior Tree is the task executor.
   The top-level tree ticks task behaviors in order:
       Init -> Perceive -> SelectArm -> SafetyCheck -> PlanPath
       -> PathSafe -> ParkInactiveArm -> PickAndPlaceWithRecovery -> Verify

2. FSM is the task-state recorder / supervisor.
   Every BT node explicitly maps to a PlannerState and writes a transition row:
       INIT -> PERCEIVE -> SELECT_ARM -> SAFETY_CHECK -> PLAN_PATH
       -> PARK_INACTIVE_ARM -> EXECUTE_PICK_PLACE -> VERIFY -> DONE/FAILED

This gives a clean story for reports:
    BT organizes behavior composition.
    FSM records lifecycle and stage transitions.
"""

from pathlib import Path
import argparse
import csv
import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import mujoco
import mujoco.viewer

TASK_PLANNER_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = TASK_PLANNER_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import task_planner.core as core
from task_planner.behavior_tree import ActionNode, BTContext, BTStatus, ConditionNode, SequenceNode


BT_TRACE_PATH = core.OUTPUT_DIR / "behavior_tree_trace_v17_rl_fallback.csv"
FSM_TRACE_PATH = core.OUTPUT_DIR / "fsm_trace_v17_rl_fallback.csv"
HYBRID_REPORT_PATH = core.OUTPUT_DIR / "hybrid_rl_fallback_report_v17.md"


def patch_v17_output_paths() -> None:
    """Route all outputs to V17-specific filenames so older successful runs are not overwritten."""
    core.LOG_PATH = core.OUTPUT_DIR / "task_log_v17_rl_fallback.csv"
    core.PLAN_PATH = core.OUTPUT_DIR / "execution_plan_v17_rl_fallback.csv"
    core.SUMMARY_PATH = core.OUTPUT_DIR / "task_summary_v17_rl_fallback.json"
    core.REPORT_PATH = core.OUTPUT_DIR / "presentation_report_v17_rl_fallback.md"
    core.STATE_MACHINE_PATH = core.OUTPUT_DIR / "state_machine_v17_rl_fallback.mmd"
    core.RUNBOOK_PATH = core.OUTPUT_DIR / "demo_runbook_v17_rl_fallback.txt"
    core.PATH_PLAN_PATH = core.OUTPUT_DIR / "path_plan_v17_rl_fallback.csv"
    core.COLLISION_LOG_PATH = core.OUTPUT_DIR / "collision_log_v17_rl_fallback.csv"


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


def init_fsm_trace() -> None:
    core.ensure_output_dir()
    with open(FSM_TRACE_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "task_id",
                "bt_node",
                "from_state",
                "to_state",
                "status",
                "message",
            ]
        )


def append_fsm_transition(task_id: str, bt_node: str, from_state: str, to_state: str, status: str, message: str) -> None:
    core.ensure_output_dir()
    with open(FSM_TRACE_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), task_id, bt_node, from_state, to_state, status, message])


@dataclass
class HybridFSM:
    """Tiny FSM tracker used together with the behavior tree.

    The actual robot execution still uses the stable core planner; this class is
    intentionally just a supervisor/trace layer so the demo clearly shows both
    FSM transitions and BT ticks.
    """

    current_state: str = "START"

    def transition(self, ctx: BTContext, bt_node: str, to_state: core.PlannerState, status: str = "ENTER", message: str = "") -> None:
        from_state = self.current_state
        self.current_state = to_state.name
        append_fsm_transition(
            task_id=ctx.task.task_id,
            bt_node=bt_node,
            from_state=from_state,
            to_state=to_state.name,
            status=status,
            message=message,
        )
        print(f"[FSM] {ctx.task.task_id}: {from_state} -> {to_state.name} via BT:{bt_node} | {message}")

    def finish(self, ctx: BTContext, bt_node: str, ok: bool, message: str = "") -> None:
        self.transition(ctx, bt_node, core.PlannerState.DONE if ok else core.PlannerState.FAILED, "DONE" if ok else "FAILED", message)


def _hybrid_action(
    name: str,
    state: core.PlannerState,
    fn: Callable[[BTContext], bool],
    message_fn: Optional[Callable[[BTContext], str]] = None,
) -> ActionNode:
    """Wrap a BT action so it also records an FSM transition before execution."""

    def wrapped(ctx: BTContext) -> bool:
        if not hasattr(ctx, "hybrid_fsm") or ctx.hybrid_fsm is None:
            ctx.hybrid_fsm = HybridFSM()
        ctx.hybrid_fsm.transition(ctx, name, state, "ENTER", f"enter {state.name}")
        return fn(ctx)

    return ActionNode(name, wrapped, message_fn)


def _hybrid_condition(
    name: str,
    state: core.PlannerState,
    predicate: Callable[[BTContext], bool],
    message_fn: Optional[Callable[[BTContext], str]] = None,
) -> ConditionNode:
    """Wrap a BT condition so it also records an FSM transition/check."""

    def wrapped(ctx: BTContext) -> bool:
        if not hasattr(ctx, "hybrid_fsm") or ctx.hybrid_fsm is None:
            ctx.hybrid_fsm = HybridFSM()
        ctx.hybrid_fsm.transition(ctx, name, state, "CHECK", f"check {state.name}")
        return predicate(ctx)

    return ConditionNode(name, wrapped, message_fn)


def build_hybrid_pick_place_tree() -> SequenceNode:
    def init_task(ctx: BTContext) -> bool:
        task = ctx.task
        print("\n" + "=" * 100)
        print(f"HYBRID FSM+BT TASK START: {task.task_id}")
        print("description:", task.description)
        print("=" * 100)
        ctx.planner.logger.log(task, core.PlannerState.INIT, "", message="hybrid task init")
        return True

    def perceive(ctx: BTContext) -> bool:
        ctx.obs = ctx.planner.perceive(ctx.task)
        ctx.planner.logger.log(ctx.task, core.PlannerState.PERCEIVE, "", message="hybrid object pose acquired")
        return "cube_pos" in ctx.obs

    def select_arm(ctx: BTContext) -> bool:
        ctx.selected_arm = ctx.planner.select_arm(ctx.obs["cube_pos"])
        ctx.planner.logger.log(ctx.task, core.PlannerState.SELECT_ARM, ctx.selected_arm, message="hybrid arm selected")
        return ctx.selected_arm in ["left", "right"]

    def safety_check(ctx: BTContext) -> bool:
        ctx.safety_snapshot = ctx.planner.safety.safety_check(ctx.selected_arm, ctx.obs["cube_pos"])
        collision_summary = ctx.planner.collision.snapshot(
            ctx.task.task_id,
            core.PlannerState.SAFETY_CHECK.name,
            ctx.selected_arm,
            message="hybrid pre-action collision snapshot",
        )
        ctx.planner.logger.log(
            ctx.task,
            core.PlannerState.SAFETY_CHECK,
            ctx.selected_arm,
            safety=ctx.safety_snapshot,
            message=f"hybrid safety pre-check | collision_dangerous={collision_summary['dangerous_count']}",
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
            message="hybrid path planning collision snapshot",
        )
        ctx.planner.logger.log(
            ctx.task,
            core.PlannerState.PLAN_PATH,
            ctx.selected_arm,
            safety=ctx.safety_snapshot,
            message=f"hybrid path planned, waypoints={len(ctx.waypoints)}, path_safe={ctx.path_safe}, collision_dangerous={collision_summary['dangerous_count']}",
        )
        return int(collision_summary["dangerous_count"]) == 0

    def park_inactive(ctx: BTContext) -> bool:
        ctx.safety_snapshot = ctx.planner.safety.park_inactive_arm(ctx.selected_arm, ctx.obs["cube_pos"])
        collision_summary = ctx.planner.collision.snapshot(
            ctx.task.task_id,
            core.PlannerState.PARK_INACTIVE_ARM.name,
            ctx.selected_arm,
            message="hybrid after inactive arm park",
        )
        ctx.planner.logger.log(
            ctx.task,
            core.PlannerState.PARK_INACTIVE_ARM,
            ctx.selected_arm,
            safety=ctx.safety_snapshot,
            message=f"hybrid inactive arm parked / safe | collision_dangerous={collision_summary['dangerous_count']}",
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
        collision_summary = ctx.planner.collision.snapshot(
            ctx.task.task_id,
            core.PlannerState.EXECUTE_PICK_PLACE.name,
            ctx.selected_arm,
            message="hybrid post-action collision snapshot",
        )
        ctx.planner.logger.log(
            ctx.task,
            core.PlannerState.EXECUTE_PICK_PLACE,
            ctx.selected_arm,
            result=ctx.result,
            safety=ctx.safety_snapshot,
            message=f"hybrid pick_and_place finished | collision_dangerous={collision_summary['dangerous_count']}",
        )
        ctx.planner.safety.release_after_task()
        return bool(ctx.result.pick_success) and int(collision_summary["dangerous_count"]) == 0

    def verify_success(ctx: BTContext) -> bool:
        ctx.planner.print_result(ctx.result)
        ok = bool(ctx.result.place_success)
        ctx.planner.logger.log(
            ctx.task,
            core.PlannerState.VERIFY,
            ctx.selected_arm,
            result=ctx.result,
            message="hybrid verify success" if ok else "hybrid verify failed",
        )
        if not hasattr(ctx, "hybrid_fsm") or ctx.hybrid_fsm is None:
            ctx.hybrid_fsm = HybridFSM()
        ctx.hybrid_fsm.finish(ctx, "VerifySuccess", ok, "hybrid task done" if ok else "hybrid task failed")
        ctx.planner.logger.log(
            ctx.task,
            core.PlannerState.DONE if ok else core.PlannerState.FAILED,
            ctx.selected_arm,
            result=ctx.result,
            message="hybrid task done" if ok else "hybrid task failed",
        )
        print("=" * 100)
        print(f"HYBRID TASK END: {ctx.task.task_id} -> {'DONE' if ok else 'FAILED'}")
        print("=" * 100)
        return ok

    return SequenceNode(
        "HybridPickPlaceTaskSequence",
        [
            _hybrid_action("InitTask", core.PlannerState.INIT, init_task),
            _hybrid_action("PerceiveObject", core.PlannerState.PERCEIVE, perceive, lambda ctx: f"cube_pos={ctx.obs['cube_pos']}"),
            _hybrid_action("SelectArm", core.PlannerState.SELECT_ARM, select_arm, lambda ctx: f"selected_arm={ctx.selected_arm}"),
            _hybrid_action("SafetyCheck", core.PlannerState.SAFETY_CHECK, safety_check, lambda ctx: f"workspace_ok={ctx.safety_snapshot.workspace_ok}"),
            _hybrid_action("PlanPath", core.PlannerState.PLAN_PATH, plan_path, lambda ctx: f"waypoints={len(ctx.waypoints)}"),
            _hybrid_condition("PathSafe", core.PlannerState.PLAN_PATH, lambda ctx: ctx.path_safe, lambda ctx: f"path_safe={ctx.path_safe}"),
            _hybrid_action("ParkInactiveArm", core.PlannerState.PARK_INACTIVE_ARM, park_inactive, lambda ctx: f"inactive_arm={ctx.safety_snapshot.inactive_arm}"),
            _hybrid_action("PickAndPlaceWithRecovery", core.PlannerState.EXECUTE_PICK_PLACE, execute_pick_place, lambda ctx: f"pick={ctx.result.pick_success}, place={ctx.result.place_success}"),
            _hybrid_condition("VerifySuccess", core.PlannerState.VERIFY, verify_success, lambda ctx: f"place_success={ctx.result.place_success}"),
        ],
    )


class HybridFSMBehaviorTreeRunner:
    def __init__(self, planner: core.BimanualTaskPlanner):
        self.planner = planner
        self.tree = build_hybrid_pick_place_tree()

    def run(self, tasks: List[core.PickPlaceTask]) -> List[core.TaskResult]:
        core.print_presentation_header(tasks)
        print("\n" + "#" * 100)
        print("V17 Hybrid FSM + BT + RL Fallback Executor")
        print("#" * 100)
        print("执行方式：")
        print("  Behavior Tree 负责 tick 行为节点。")
        print("  FSM 负责记录每个节点对应的任务阶段和状态转移。")
        print("\n行为树：")
        print("  Sequence(HybridPickPlaceTask)")
        print("    -> InitTask")
        print("    -> PerceiveObject")
        print("    -> SelectArm")
        print("    -> SafetyCheck")
        print("    -> PlanPath")
        print("    -> PathSafe")
        print("    -> ParkInactiveArm")
        print("    -> PickAndPlaceWithRecovery")
        print("    -> VerifySuccess")
        print("\n状态机：")
        print("  INIT -> PERCEIVE -> SELECT_ARM -> SAFETY_CHECK -> PLAN_PATH")
        print("       -> PARK_INACTIVE_ARM -> EXECUTE_PICK_PLACE -> VERIFY -> DONE/FAILED")
        print("#" * 100)

        results: List[core.TaskResult] = []
        for i, task in enumerate(tasks, start=1):
            print(f"\n[HYBRID RUNNER] Running task {i}/{len(tasks)}: {task.task_id}")
            ctx = BTContext(task=task, planner=self.planner)
            # Attach FSM tracker dynamically; this keeps behavior_tree.py dependency-free.
            ctx.hybrid_fsm = HybridFSM()
            status = self.tree.tick(ctx)
            append_bt_trace(ctx)
            if ctx.result is not None:
                results.append(ctx.result)
            else:
                result = self.planner.make_failure_result(task, ctx.selected_arm or "", message=f"Hybrid BT/FSM failed before action: {status.name}")
                results.append(result)
                append_fsm_transition(task.task_id, "HybridRunner", ctx.hybrid_fsm.current_state, core.PlannerState.FAILED.name, "FAILED", result.message)

            if i < len(tasks):
                print("[HYBRID RUNNER] pause 1.5s before next task")
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
        write_hybrid_report(tasks, results)
        print("BT_TRACE_PATH:", BT_TRACE_PATH)
        print("FSM_TRACE_PATH:", FSM_TRACE_PATH)
        print("HYBRID_REPORT_PATH:", HYBRID_REPORT_PATH)
        return results


def write_hybrid_report(tasks: List[core.PickPlaceTask], results: List[core.TaskResult]) -> None:
    success_count = sum(1 for r in results if r.place_success)
    core.ensure_output_dir()
    with open(HYBRID_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# OpenArm V17 Hybrid FSM + BT + RL Fallback Demo\n\n")
        f.write("## Purpose\n\n")
        f.write("This demo combines a Behavior Tree task executor, a finite-state machine trace, and a deployable right-arm RL adapter with rule-expert fallback. ")
        f.write("The Behavior Tree organizes task behaviors, while the FSM records the lifecycle states of each task.\n\n")
        f.write("## Behavior Tree\n\n")
        f.write("```text\n")
        f.write("Sequence(HybridPickPlaceTask)\n")
        f.write("  -> InitTask\n")
        f.write("  -> PerceiveObject\n")
        f.write("  -> SelectArm\n")
        f.write("  -> SafetyCheck\n")
        f.write("  -> PlanPath\n")
        f.write("  -> PathSafe\n")
        f.write("  -> ParkInactiveArm\n")
        f.write("  -> PickAndPlaceWithRecovery\n")
        f.write("  -> VerifySuccess\n")
        f.write("```\n\n")
        f.write("## FSM States\n\n")
        f.write("```text\n")
        f.write("INIT -> PERCEIVE -> SELECT_ARM -> SAFETY_CHECK -> PLAN_PATH -> PARK_INACTIVE_ARM -> EXECUTE_PICK_PLACE -> VERIFY -> DONE/FAILED\n")
        f.write("```\n\n")
        f.write("## Results\n\n")
        f.write(f"Total place_success: {success_count}/{len(results)}\n\n")
        f.write("| Task | Arm | Pick | Place | Lift | XY | Z margin |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        for r in results:
            f.write(
                f"| {r.task_id} | {r.selected_arm} | {r.pick_success} | {r.place_success} | "
                f"{r.lift_delta:.4f} | {r.xy_dist:.4f} | {r.z_margin:.4f} |\n"
            )
        f.write("\n## Generated Logs\n\n")
        f.write(f"- Task log: `{core.LOG_PATH}`\n")
        f.write(f"- Behavior-tree trace: `{BT_TRACE_PATH}`\n")
        f.write(f"- FSM trace: `{FSM_TRACE_PATH}`\n")
        f.write(f"- Collision log: `{core.COLLISION_LOG_PATH}`\n")


def parse_args():
    parser = argparse.ArgumentParser(description="OpenArm V13 hybrid FSM + behavior-tree task planner demo")
    parser.add_argument("--no-viewer", action="store_true", help="不打开 viewer，快速 headless 测试")
    parser.add_argument("--right-only", action="store_true", help="只运行右臂稳定展示任务")
    parser.add_argument("--queue-demo", action="store_true", help="运行 4 个任务的队列演示：left/right/left/right")
    parser.add_argument("--simulate-recovery", action="store_true", help="在第一个任务第一次尝试时注入一次模拟失败，展示 retry 恢复逻辑")
    parser.add_argument("--max-retries", type=int, default=1, help="失败后同一只手最多重试次数，默认 1")
    parser.add_argument("--no-fallback", dest="enable_fallback", action="store_false", help="失败后不切换到另一只手兜底")
    parser.add_argument("--no-hold-viewer", dest="hold_viewer", action="store_false", help="结束后不保持 viewer")
    parser.add_argument("--right-adapter", choices=["rule", "rl", "rl_fallback"], default="rule", help="右臂执行器：rule / rl / rl_fallback。默认 rule，推荐展示 RL 接入用 rl_fallback")
    parser.add_argument("--rl-model", type=str, default=str(core.ROOT / "outputs" / "rl_right_pick_place_v16_reward_fix" / "ppo_right_pick_place_v16_final.zip"), help="右臂 RL PPO 模型路径")
    parser.add_argument("--rl-vecnormalize", type=str, default=str(core.ROOT / "outputs" / "rl_right_pick_place_v16_reward_fix" / "vecnormalize_v16_final.pkl"), help="右臂 RL VecNormalize 路径")
    parser.add_argument("--rl-max-steps", type=int, default=450, help="RL policy 单次尝试最大 step 数")
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
    init_fsm_trace()

    planner = core.BimanualTaskPlanner(
        model,
        data,
        viewer=viewer,
        realtime=realtime,
        max_retries=args.max_retries,
        enable_fallback=args.enable_fallback,
        simulate_recovery=args.simulate_recovery,
        right_adapter_mode=args.right_adapter,
        rl_model_path=args.rl_model,
        rl_vecnormalize_path=args.rl_vecnormalize,
        rl_max_steps=args.rl_max_steps,
    )
    runner = HybridFSMBehaviorTreeRunner(planner)
    runner.run(tasks)


def main() -> None:
    args = parse_args()
    patch_v17_output_paths()
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
        print("viewer 已打开。2 秒后开始 V17 FSM+BT+RL fallback 混合任务规划 demo。")
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

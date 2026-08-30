from __future__ import annotations

"""Official-YCB + strict-YOLOv8-Seg MuJoCo upper-computer bridge.

Unlike the legacy multi-object classroom demo, this profile does not move a
selected target to a fixed loading station.  The right arm approaches the
object at its current scattered table position.  The visual stream is
published continuously while the planner runs.
"""

import argparse
import asyncio
from pathlib import Path
import random
import sys
import time
from typing import Any

import mujoco
import numpy as np
import websockets


VISION_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = VISION_DIR.parent
ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from vision.multi_object_detector import (  # noqa: E402
    DEFAULT_SEG_WEIGHTS,
    SimulationObjectSpec,
)
from vision.mujoco_upper_bridge import (  # noqa: E402
    CAMERA_NAME,
    MujocoUpperBridge,
    patch_bridge_output_paths,
)
import task_planner.core as core  # noqa: E402


YCB_XML = ROOT / "v2" / "demo_ycb_real_objects.xml"
YCB_SPAWNS = {
    "ycb_bottle": np.array([0.580, 0.050, 1.006], dtype=float),
    "ycb_banana": np.array([0.490, 0.030, 1.006], dtype=float),
    "ycb_apple": np.array([0.550, 0.000, 1.006], dtype=float),
    "ycb_cup": np.array([0.450, -0.090, 1.006], dtype=float),
}

# Four deliberately separated layouts inside the calibrated right-arm strip.
# The shuffle button chooses a layout different from the current one.  These
# are finite rather than arbitrary coordinates so every object/layout pair can
# be regression-tested before classroom use.
YCB_SHUFFLE_LAYOUTS = (
    {
        "ycb_bottle": np.array([0.580, 0.050, 1.006], dtype=float),
        "ycb_banana": np.array([0.490, 0.030, 1.006], dtype=float),
        "ycb_apple": np.array([0.550, 0.000, 1.006], dtype=float),
        "ycb_cup": np.array([0.450, -0.090, 1.006], dtype=float),
    },
    {
        "ycb_bottle": np.array([0.585, 0.055, 1.006], dtype=float),
        "ycb_banana": np.array([0.490, 0.030, 1.006], dtype=float),
        "ycb_apple": np.array([0.550, 0.000, 1.006], dtype=float),
        "ycb_cup": np.array([0.445, -0.095, 1.006], dtype=float),
    },
    {
        "ycb_bottle": np.array([0.580, 0.055, 1.006], dtype=float),
        "ycb_banana": np.array([0.490, 0.030, 1.006], dtype=float),
        "ycb_apple": np.array([0.552, 0.002, 1.006], dtype=float),
        "ycb_cup": np.array([0.455, -0.095, 1.006], dtype=float),
    },
    {
        "ycb_bottle": np.array([0.585, 0.050, 1.006], dtype=float),
        "ycb_banana": np.array([0.487, 0.033, 1.006], dtype=float),
        "ycb_apple": np.array([0.547, 0.003, 1.006], dtype=float),
        "ycb_cup": np.array([0.440, -0.090, 1.006], dtype=float),
    },
)

# Additional per-layout camera-centroid corrections measured from strict
# YOLOv8-Seg renders.  Layout 1 is the original class calibration baseline.
# These are calibration residuals, not runtime body-position reads.
YCB_LAYOUT_XY_CORRECTIONS = (
    {},
    {
        "ycb_bottle": np.array([-0.00184, 0.00534]),
        "ycb_banana": np.array([0.00012, -0.00045]),
        "ycb_apple": np.array([-0.00235, -0.00480]),
        "ycb_cup": np.array([-0.00093, -0.00190]),
    },
    {
        "ycb_bottle": np.array([0.00053, 0.00340]),
        "ycb_banana": np.array([0.00026, -0.00072]),
        "ycb_apple": np.array([-0.00370, -0.00740]),
        "ycb_cup": np.array([0.00120, 0.00024]),
    },
    {
        "ycb_bottle": np.array([-0.00218, 0.00650]),
        "ycb_banana": np.array([0.00319, 0.00120]),
        "ycb_apple": np.array([0.00161, -0.00101]),
        "ycb_cup": np.array([0.00111, 0.00078]),
    },
)

YCB_OBJECT_SPECS = (
    SimulationObjectSpec(
        "ycb_bottle", "bottle", "YCB 芥末瓶", 1.006, 1.006, "unused",
        ("bottle",), 1.046, (-0.00476, -0.00633),
    ),
    SimulationObjectSpec(
        "ycb_banana", "banana", "YCB 香蕉", 1.006, 1.006, "unused",
        ("banana",), 1.013, (-0.00212, -0.00021),
    ),
    SimulationObjectSpec(
        "ycb_apple", "apple", "YCB 苹果", 1.006, 1.006, "unused",
        ("apple",), 1.017, (-0.00214, -0.00268),
    ),
    SimulationObjectSpec(
        "ycb_cup", "cup", "YCB 杯子", 1.006, 1.006, "unused",
        ("cup",), 1.020, (-0.00125, 0.00375),
    ),
)


def _rule_config(name: str, grasp_z: float) -> dict[str, Any]:
    grasp = np.array([-0.006, -0.020, grasp_z], dtype=float)
    return {
        "name": f"ycb_direct_{name}_z{grasp_z:.3f}",
        "site_type": "tcp",
        "pregrasp_offset": grasp + np.array([0.0, 0.0, 0.085]),
        "grasp_offset": grasp,
        "preplace_offset": np.array([0.0, 0.0, 0.120]),
        "place_offset": np.array([0.0, 0.0, 0.050]),
        "joint_biases": {"right_joint7_ctrl": -0.060},
    }


YCB_RULE_CONFIGS = {
    "ycb_bottle": _rule_config("bottle", 0.075),
    "ycb_banana": _rule_config("banana", 0.055),
    "ycb_apple": _rule_config("apple", 0.055),
    "ycb_cup": _rule_config("cup", 0.055),
}


def prepare_ycb_scene(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    viewer: Any = None,
    realtime: bool = False,
    spawns: dict[str, np.ndarray] | None = None,
) -> dict[str, list[float]]:
    selected_spawns = YCB_SPAWNS if spawns is None else spawns
    if set(selected_spawns) != set(YCB_SPAWNS):
        raise ValueError("YCB layout must contain exactly the four supported bodies")
    core.load_home(model, data)
    for actuator, value in (
        # Both arms start up/outboard so the camera can see the scatter area.
        # The selected right arm later returns along a visible ready trajectory.
        ("left_joint1_ctrl", -1.00),
        ("left_joint2_ctrl", -0.50),
        ("left_joint3_ctrl", 0.00),
        ("left_joint4_ctrl", 1.20),
        ("right_joint1_ctrl", 1.00),
        ("right_joint2_ctrl", 0.50),
        ("right_joint3_ctrl", 0.00),
        ("right_joint4_ctrl", 1.20),
        ("left_finger1_ctrl", core.get_left_attr("LEFT_FINGER_PRE_OPEN", 0.445)),
        ("right_finger1_ctrl", core.right_rule.RIGHT_FINGER_PRE_OPEN),
        ("lifter_ctrl", 0.0),
    ):
        if core.maybe_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator) != -1:
            core.set_ctrl(model, data, actuator, value)
    mujoco.mj_forward(model, data)
    # Park before spawning the loose objects so the outgoing grippers cannot
    # sweep them across the table during reset.
    core.sim_steps(model, data, 500, viewer=viewer, realtime=realtime)
    for body_name, position in selected_spawns.items():
        core.set_free_body_pos(model, data, body_name, position)
    mujoco.mj_forward(model, data)
    core.sim_steps(model, data, 240, viewer=viewer, realtime=realtime)
    return {
        name: [float(value) for value in core.get_body_pos(model, data, name)]
        for name in selected_spawns
    }


class YCBUpperBridge(MujocoUpperBridge):
    def __init__(self, *args: Any, speed_scale: float = 0.40, **kwargs: Any) -> None:
        # This calibrated profile assigns the whole x=0.45..0.58,
        # y=-0.09..0.05 strip to the right arm.  Keep the legacy threshold
        # unchanged in other processes/profiles.
        core.SELECT_RIGHT_IF_CUBE_Y_GE = -0.10
        super().__init__(
            *args,
            detector_backend="yolo",
            yolo_weights=DEFAULT_SEG_WEIGHTS,
            direct_pose_execution=True,
            object_rule_configs=YCB_RULE_CONFIGS,
            rule_speed_scale=speed_scale,
            detector_specs=YCB_OBJECT_SPECS,
            **kwargs,
        )
        self._shuffle_rng = random.SystemRandom()
        self._current_layout_index = 0

    def postprocess_detections(self, detections: Any) -> tuple[Any, ...]:
        correction_by_body = YCB_LAYOUT_XY_CORRECTIONS[self._current_layout_index]
        corrected = tuple(detections)
        for detection in corrected:
            correction = correction_by_body.get(detection.body_name)
            if correction is None:
                continue
            detection.stable_world_pos[:2] += correction
            detection.planner_cube_pos[:2] += correction
        return corrected

    def randomize_scene(self) -> dict[str, Any]:
        candidates = [
            index
            for index in range(len(YCB_SHUFFLE_LAYOUTS))
            if index != self._current_layout_index
        ]
        next_index = self._shuffle_rng.choice(candidates)
        stable = prepare_ycb_scene(
            self.model,
            self.data,
            viewer=self.planner_viewer,
            realtime=self.realtime,
            spawns=YCB_SHUFFLE_LAYOUTS[next_index],
        )
        self._current_layout_index = next_index
        return {"layout_index": next_index, "positions": stable}

    def run_home(self) -> None:
        prepare_ycb_scene(
            self.model,
            self.data,
            viewer=self.planner_viewer,
            realtime=self.realtime,
        )

    def prepare_direct_execution(self) -> None:
        """Move the active arm from camera-clear standby to its proven IK seed."""
        core.move_to_ctrl(
            self.model,
            self.data,
            {
                "right_joint1_ctrl": 0.0,
                "right_joint2_ctrl": 0.0,
                "right_joint3_ctrl": 0.0,
                "right_joint4_ctrl": 1.570796,
                "right_joint5_ctrl": 0.0,
                "right_joint6_ctrl": 0.0,
                "right_joint7_ctrl": 0.0,
                "right_finger1_ctrl": core.right_rule.RIGHT_FINGER_PRE_OPEN,
            },
            duration=max(0.55, 1.20 * self.rule_speed_scale),
            viewer=self.planner_viewer,
            realtime=self.realtime,
        )
        core.sim_steps(
            self.model,
            self.data,
            140,
            viewer=self.planner_viewer,
            realtime=self.realtime,
        )


async def serve_bridge(args: argparse.Namespace, viewer: Any = None) -> None:
    patch_bridge_output_paths()
    core.ensure_output_dir()
    model = mujoco.MjModel.from_xml_path(str(YCB_XML))
    data = mujoco.MjData(model)
    stable = prepare_ycb_scene(model, data, viewer=viewer, realtime=True)
    bridge = YCBUpperBridge(
        model,
        data,
        fps=args.fps,
        viewer=viewer,
        realtime=True,
        yolo_confidence=args.confidence,
        speed_scale=args.speed_scale,
        realtime_playback_rate=args.playback_rate,
    )

    print("=" * 76)
    print("OpenArm YCB Real-Object Simulation Upper-Computer Bridge")
    print(f"WebSocket: ws://{args.host}:{args.port}")
    print(f"Scene: {YCB_XML}")
    print(f"Objects: {stable}")
    print(f"Perception: {bridge.detector.status}")
    print("Execution: direct scattered pose; no loading-station teleport")
    print(f"Motion speed scale: {args.speed_scale:.2f}")
    print(f"Viewer playback rate: {args.playback_rate:.2f}x")
    print("Pose scope: tabletop XY + mask image angle; not full 6D pose")
    print("=" * 76)

    async with websockets.serve(
        bridge.handler,
        args.host,
        args.port,
        max_size=4_000_000,
        ping_interval=10,
        ping_timeout=20,
    ):
        interval = 1.0 / bridge.fps
        while viewer is None or viewer.is_running():
            started = time.monotonic()
            await bridge.process_one_command()
            published_live = await bridge.publish_pending_live_frame()
            await bridge.publish_progress_heartbeat()
            if not bridge.command_active:
                for _ in range(max(1, int(0.01 / model.opt.timestep))):
                    mujoco.mj_step(model, data)
                bridge.sync_viewer()
                if not published_live:
                    await bridge.publish_once()
            await asyncio.sleep(max(0.0, interval - (time.monotonic() - started)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official-YCB strict-YOLOv8-Seg bridge")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--confidence", type=float, default=0.06)
    parser.add_argument("--speed-scale", type=float, default=0.40)
    parser.add_argument("--playback-rate", type=float, default=1.80)
    parser.add_argument("--viewer", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.viewer:
        asyncio.run(serve_bridge(args))
        return
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(YCB_XML))
    data = mujoco.MjData(model)
    viewer_data = mujoco.MjData(model)
    # Finish reset before opening the viewer.  Physics then runs only on
    # ``data`` while OpenGL displays snapshots copied into ``viewer_data``.
    # This avoids concurrent access to a single MjData from two threads.
    patch_bridge_output_paths()
    core.ensure_output_dir()
    prepare_ycb_scene(model, data, viewer=None, realtime=False)
    bridge = YCBUpperBridge(
        model,
        data,
        fps=args.fps,
        viewer=None,
        viewer_data=viewer_data,
        realtime=True,
        yolo_confidence=args.confidence,
        speed_scale=args.speed_scale,
        realtime_playback_rate=args.playback_rate,
    )
    with mujoco.viewer.launch_passive(model, viewer_data) as viewer:
        bridge.set_viewer(viewer, viewer_data)
        bridge.sync_viewer()

        async def viewer_loop() -> None:
            print(f"[YCB bridge] WebSocket ws://{args.host}:{args.port}; {bridge.detector.status}")
            async with websockets.serve(
                bridge.handler,
                args.host,
                args.port,
                max_size=4_000_000,
                ping_interval=10,
                ping_timeout=20,
            ):
                interval = 1.0 / bridge.fps
                while viewer.is_running():
                    started = time.monotonic()
                    await bridge.process_one_command()
                    published_live = await bridge.publish_pending_live_frame()
                    await bridge.publish_progress_heartbeat()
                    if not bridge.command_active:
                        for _ in range(max(1, int(0.01 / model.opt.timestep))):
                            mujoco.mj_step(model, data)
                        bridge.sync_viewer()
                        if not published_live:
                            await bridge.publish_once()
                    await asyncio.sleep(max(0.0, interval - (time.monotonic() - started)))

        asyncio.run(viewer_loop())


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
import os
import shutil
import textwrap
import zipfile


V18_ROOT = Path.cwd()
V19_ROOT = V18_ROOT.parent / "openarm_v19_ros2_migration"
ZIP_PATH = V18_ROOT.parent / "openarm_v19_ros2_migration_full.zip"


def write(rel_path: str, content: str) -> None:
    path = V19_ROOT / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
    print("[WRITE]", path.relative_to(V19_ROOT))


def copy_dir(name: str) -> None:
    src = V18_ROOT / name
    dst = V19_ROOT / "v18_snapshot" / name

    if not src.exists():
        print("[SKIP]", src)
        return

    shutil.copytree(
        src,
        dst,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            ".git",
            ".idea",
            ".vscode",
            "outputs",
            "_archive_unused*",
        ),
    )
    print("[COPY]", src, "->", dst)


def create_snapshot() -> None:
    for name in ["v2", "scripts", "src"]:
        copy_dir(name)

    for pattern in [
        "README*",
        "requirements*.txt",
        "pyproject.toml",
        "setup.py",
    ]:
        for src in V18_ROOT.glob(pattern):
            if src.is_file():
                dst = V19_ROOT / "v18_snapshot" / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)


def create_common_files() -> None:
    write(
        "VERSION",
        """
        19.0.0
        """,
    )

    write(
        "README.md",
        r"""
        # OpenArm V19 ROS2 Migration

        本工程与原 MuJoCo 仿真工程分开：

        ```text
        E:\FL_Personal\
        ├── openarm_mujoco-master
        ├── openarm_v19_ros2_migration
        └── openarm_v19_ros2_migration_full.zip
        ```

        ## 架构

        ```text
        V18 MuJoCo 仿真
                ↓
        Simulation Bridge
                ↓
        Arm Interface
           /           \
        MuJoCo         ROS2
                         ↓
                  openarm_ros2
                         ↓
                  ros2_control
                         ↓
                  openarm_can
                         ↓
                    真机电机
        ```

        ## Windows 仿真检查

        ```bat
        python windows_sim_bridge\run_v19.py --check
        ```

        通过 V19 运行原视觉抓取：

        ```bat
        python windows_sim_bridge\run_v19.py --vision
        ```

        双臂视觉抓取：

        ```bat
        python windows_sim_bridge\run_v19.py --vision --both-arms
        ```

        ## Ubuntu ROS2

        ```bash
        bash scripts/install_openarm_vendor.sh
        bash scripts/build_ros2_ws.sh

        source /opt/ros/$ROS_DISTRO/setup.bash
        source ros2_ws/install/setup.bash

        ros2 launch openarm_v19_bridge openarm_v19_bridge.launch.py
        ```

        默认 `dry_run: true`，不会直接让真机运动。
        """,
    )

    write(
        "config/openarm_v19.yaml",
        """
        version: "19.0.0"

        planner:
          arm_selection:
            strategy: "y_threshold"
            y_threshold_m: 0.025
          max_retries: 1
          cross_arm_fallback: false

        frames:
          world: "world"
          base: "base_link"
          camera: "camera_color_optical_frame"
          left_tcp: "left_tcp"
          right_tcp: "right_tcp"

        safety:
          dry_run: true
          velocity_scale: 0.10
          acceleration_scale: 0.10
          minimum_inter_arm_distance_m: 0.18

          workspace:
            x_min: 0.20
            x_max: 0.75
            y_min: -0.45
            y_max: 0.45
            z_min: 0.75
            z_max: 1.45

        ros2:
          task_topic: "/openarm/task_command"
          status_topic: "/openarm/task_status"
          left_action: "/left_arm_controller/follow_joint_trajectory"
          right_action: "/right_arm_controller/follow_joint_trajectory"
        """,
    )


def create_arm_interface() -> None:
    write("arm_interface/__init__.py", "")

    write(
        "arm_interface/base_arm.py",
        '''
        from __future__ import annotations

        from dataclasses import dataclass
        from enum import Enum
        from typing import Optional, Tuple


        class ArmName(str, Enum):
            LEFT = "left"
            RIGHT = "right"


        @dataclass
        class PoseTarget:
            position: Tuple[float, float, float]
            frame_id: str = "base_link"
            orientation_xyzw: Optional[
                Tuple[float, float, float, float]
            ] = None


        @dataclass
        class PickPlaceCommand:
            task_id: str
            object_name: str
            target_name: str
            object_pose: PoseTarget
            requested_arm: Optional[ArmName] = None


        @dataclass
        class ExecutionResult:
            task_id: str
            arm: ArmName
            success: bool
            backend: str
            message: str


        class BaseArmController:
            def connect(self) -> bool:
                raise NotImplementedError

            def move_pose(self, target: PoseTarget) -> bool:
                raise NotImplementedError

            def open_gripper(self) -> bool:
                raise NotImplementedError

            def close_gripper(self) -> bool:
                raise NotImplementedError

            def pick_and_place(
                self,
                command: PickPlaceCommand,
            ) -> ExecutionResult:
                raise NotImplementedError
        ''',
    )

    write(
        "arm_interface/mujoco_backend.py",
        '''
        from __future__ import annotations

        import subprocess
        import sys
        from pathlib import Path


        class MujocoV18Backend:
            """调用 V19 内保存的 V18 稳定仿真快照。"""

            def __init__(self, v19_root: Path):
                self.v19_root = Path(v19_root)
                self.snapshot = self.v19_root / "v18_snapshot"
                self.demo = (
                    self.snapshot
                    / "scripts"
                    / "task_planner"
                    / "run_vision_grasp_demo.py"
                )

            def connect(self) -> bool:
                return self.demo.exists()

            def run_vision_demo(
                self,
                both_arms: bool = False,
                no_viewer: bool = False,
            ) -> int:
                if not self.demo.exists():
                    raise FileNotFoundError(self.demo)

                command = [sys.executable, str(self.demo)]

                if both_arms:
                    command.append("--both-arms")

                if no_viewer:
                    command.append("--no-viewer")

                print("[V19 MUJOCO]", subprocess.list2cmdline(command))

                result = subprocess.run(
                    command,
                    cwd=str(self.snapshot),
                    check=False,
                )
                return int(result.returncode)
        ''',
    )

    write(
        "arm_interface/ros2_backend.py",
        '''
        from __future__ import annotations

        from .base_arm import (
            ArmName,
            BaseArmController,
            ExecutionResult,
            PickPlaceCommand,
            PoseTarget,
        )


        class ROS2ArmController(BaseArmController):
            """真机控制后端。

            V19.0 建立接口；
            实际轨迹由 ROS2 Bridge、MoveIt2 和 ros2_control 执行。
            """

            def __init__(
                self,
                arm: ArmName,
                dry_run: bool = True,
            ):
                self.arm = arm
                self.dry_run = dry_run
                self.connected = False

            def connect(self) -> bool:
                try:
                    import rclpy  # noqa: F401
                    self.connected = True
                except ImportError:
                    self.connected = False
                return self.connected

            def move_pose(self, target: PoseTarget) -> bool:
                if self.dry_run:
                    print("[DRY RUN]", self.arm.value, target)
                    return True

                raise NotImplementedError(
                    "MoveIt2 pose execution is implemented in V19.2"
                )

            def open_gripper(self) -> bool:
                if self.dry_run:
                    print("[DRY RUN] open", self.arm.value, "gripper")
                    return True
                raise NotImplementedError

            def close_gripper(self) -> bool:
                if self.dry_run:
                    print("[DRY RUN] close", self.arm.value, "gripper")
                    return True
                raise NotImplementedError

            def pick_and_place(
                self,
                command: PickPlaceCommand,
            ) -> ExecutionResult:
                return ExecutionResult(
                    task_id=command.task_id,
                    arm=self.arm,
                    success=False,
                    backend="ros2",
                    message="V19.0 bridge ready; real motion disabled",
                )
        ''',
    )


def create_windows_bridge() -> None:
    write(
        "windows_sim_bridge/run_v19.py",
        '''
        from __future__ import annotations

        from pathlib import Path
        import argparse
        import json
        import sys

        ROOT = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(ROOT))

        from arm_interface.mujoco_backend import MujocoV18Backend


        def main():
            parser = argparse.ArgumentParser(
                description="OpenArm V19 standalone bridge"
            )
            parser.add_argument("--check", action="store_true")
            parser.add_argument("--vision", action="store_true")
            parser.add_argument("--both-arms", action="store_true")
            parser.add_argument("--no-viewer", action="store_true")
            args = parser.parse_args()

            backend = MujocoV18Backend(ROOT)

            result = {
                "v19_root": str(ROOT),
                "snapshot": str(backend.snapshot),
                "vision_demo": str(backend.demo),
                "connected": backend.connect(),
            }

            print("=" * 90)
            print("OpenArm V19 Simulation Bridge")
            print("=" * 90)
            print(json.dumps(result, ensure_ascii=False, indent=2))

            output = ROOT / "outputs" / "v19_environment_check.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print("[V19] report:", output)

            if args.vision:
                if not backend.connect():
                    raise SystemExit(
                        "V18 snapshot vision demo was not found."
                    )

                code = backend.run_vision_demo(
                    both_arms=args.both_arms,
                    no_viewer=args.no_viewer,
                )
                raise SystemExit(code)

            if not args.check:
                print("使用 --check 或 --vision。")


        if __name__ == "__main__":
            main()
        ''',
    )

    write(
        "scripts/run_v19_check.bat",
        r"""
        @echo off
        cd /d %~dp0\..
        python windows_sim_bridge\run_v19.py --check
        pause
        """,
    )

    write(
        "scripts/run_v19_vision.bat",
        r"""
        @echo off
        cd /d %~dp0\..
        python windows_sim_bridge\run_v19.py --vision
        pause
        """,
    )


def create_ros2_package() -> None:
    package = "ros2_ws/src/openarm_v19_bridge"

    write(
        f"{package}/package.xml",
        """<?xml version="1.0"?>
        <package format="3">
          <name>openarm_v19_bridge</name>
          <version>19.0.0</version>
          <description>
            OpenArm V19 MuJoCo to ROS2 migration bridge
          </description>

          <maintainer email="student@example.com">
            OpenArm Software Team
          </maintainer>

          <license>Apache-2.0</license>

          <buildtool_depend>ament_python</buildtool_depend>

          <exec_depend>rclpy</exec_depend>
          <exec_depend>std_msgs</exec_depend>
          <exec_depend>geometry_msgs</exec_depend>
          <exec_depend>trajectory_msgs</exec_depend>
          <exec_depend>control_msgs</exec_depend>
          <exec_depend>tf2_ros</exec_depend>
          <exec_depend>tf2_geometry_msgs</exec_depend>

          <export>
            <build_type>ament_python</build_type>
          </export>
        </package>
        """,
    )

    write(
        f"{package}/setup.py",
        '''
        from setuptools import find_packages, setup
        from glob import glob
        import os

        package_name = "openarm_v19_bridge"

        setup(
            name=package_name,
            version="19.0.0",
            packages=find_packages(),
            data_files=[
                (
                    "share/ament_index/resource_index/packages",
                    ["resource/" + package_name],
                ),
                (
                    "share/" + package_name,
                    ["package.xml"],
                ),
                (
                    os.path.join("share", package_name, "launch"),
                    glob("launch/*.launch.py"),
                ),
                (
                    os.path.join("share", package_name, "config"),
                    glob("config/*.yaml"),
                ),
            ],
            install_requires=["setuptools"],
            zip_safe=True,
            maintainer="OpenArm Software Team",
            maintainer_email="student@example.com",
            description="OpenArm V19 ROS2 bridge",
            license="Apache-2.0",
            entry_points={
                "console_scripts": [
                    (
                        "task_bridge = "
                        "openarm_v19_bridge.task_bridge:main"
                    ),
                ],
            },
        )
        ''',
    )

    write(
        f"{package}/setup.cfg",
        """
        [develop]
        script_dir=$base/lib/openarm_v19_bridge

        [install]
        install_scripts=$base/lib/openarm_v19_bridge
        """,
    )

    write(
        f"{package}/resource/openarm_v19_bridge",
        "",
    )

    write(
        f"{package}/openarm_v19_bridge/__init__.py",
        '__version__ = "19.0.0"\n',
    )

    write(
        f"{package}/openarm_v19_bridge/task_bridge.py",
        '''
        from __future__ import annotations

        import json

        import rclpy
        from rclpy.action import ActionClient
        from rclpy.node import Node
        from std_msgs.msg import String
        from control_msgs.action import FollowJointTrajectory


        class TaskBridge(Node):
            def __init__(self):
                super().__init__(
                    "openarm_v19_task_bridge"
                )

                self.declare_parameter("dry_run", True)
                self.declare_parameter(
                    "y_threshold_m",
                    0.025,
                )
                self.declare_parameter(
                    "left_action",
                    (
                        "/left_arm_controller/"
                        "follow_joint_trajectory"
                    ),
                )
                self.declare_parameter(
                    "right_action",
                    (
                        "/right_arm_controller/"
                        "follow_joint_trajectory"
                    ),
                )

                self.dry_run = bool(
                    self.get_parameter("dry_run").value
                )
                self.y_threshold = float(
                    self.get_parameter(
                        "y_threshold_m"
                    ).value
                )

                self.left_client = ActionClient(
                    self,
                    FollowJointTrajectory,
                    str(
                        self.get_parameter(
                            "left_action"
                        ).value
                    ),
                )
                self.right_client = ActionClient(
                    self,
                    FollowJointTrajectory,
                    str(
                        self.get_parameter(
                            "right_action"
                        ).value
                    ),
                )

                self.status_pub = self.create_publisher(
                    String,
                    "/openarm/task_status",
                    10,
                )

                self.task_sub = self.create_subscription(
                    String,
                    "/openarm/task_command",
                    self.on_task,
                    10,
                )

                self.get_logger().info(
                    "OpenArm V19 bridge ready. "
                    f"dry_run={self.dry_run}"
                )

            def publish_status(self, payload):
                message = String()
                message.data = json.dumps(
                    payload,
                    ensure_ascii=False,
                )
                self.status_pub.publish(message)
                self.get_logger().info(message.data)

            def on_task(self, message):
                try:
                    task = json.loads(message.data)
                    position = task["position"]

                    x = float(position[0])
                    y = float(position[1])
                    z = float(position[2])

                    if not (
                        0.20 <= x <= 0.75
                        and -0.45 <= y <= 0.45
                        and 0.75 <= z <= 1.45
                    ):
                        self.publish_status({
                            "task_id": task.get(
                                "task_id",
                                "unknown",
                            ),
                            "state": "REJECTED",
                            "reason": (
                                "outside safety workspace"
                            ),
                        })
                        return

                    selected_arm = (
                        "right"
                        if y > self.y_threshold
                        else "left"
                    )

                    client = (
                        self.right_client
                        if selected_arm == "right"
                        else self.left_client
                    )

                    server_ready = (
                        True
                        if self.dry_run
                        else client.wait_for_server(
                            timeout_sec=2.0
                        )
                    )

                    self.publish_status({
                        "task_id": task.get(
                            "task_id",
                            "unknown",
                        ),
                        "state": "BRIDGE_READY",
                        "selected_arm": selected_arm,
                        "dry_run": self.dry_run,
                        "trajectory_server_ready": (
                            server_ready
                        ),
                        "position": [x, y, z],
                        "next": (
                            "MoveIt2 IK and trajectory"
                        ),
                    })

                except Exception as error:
                    self.publish_status({
                        "state": "ERROR",
                        "message": str(error),
                    })


        def main(args=None):
            rclpy.init(args=args)
            node = TaskBridge()

            try:
                rclpy.spin(node)
            finally:
                node.destroy_node()
                rclpy.shutdown()


        if __name__ == "__main__":
            main()
        ''',
    )

    write(
        f"{package}/config/openarm_v19.yaml",
        """
        openarm_v19_task_bridge:
          ros__parameters:
            dry_run: true
            y_threshold_m: 0.025

            left_action: >
              /left_arm_controller/follow_joint_trajectory

            right_action: >
              /right_arm_controller/follow_joint_trajectory
        """,
    )

    write(
        f"{package}/launch/openarm_v19_bridge.launch.py",
        '''
        from launch import LaunchDescription
        from launch_ros.actions import Node
        from ament_index_python.packages import (
            get_package_share_directory,
        )
        import os


        def generate_launch_description():
            share = get_package_share_directory(
                "openarm_v19_bridge"
            )

            config = os.path.join(
                share,
                "config",
                "openarm_v19.yaml",
            )

            return LaunchDescription([
                Node(
                    package="openarm_v19_bridge",
                    executable="task_bridge",
                    name="openarm_v19_task_bridge",
                    parameters=[config],
                    output="screen",
                )
            ])
        ''',
    )


def create_vendor_scripts() -> None:
    write(
        "scripts/install_openarm_vendor.sh",
        r'''#!/usr/bin/env bash
        set -euo pipefail

        ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
        SRC="$ROOT/ros2_ws/src"

        mkdir -p "$SRC"
        cd "$SRC"

        clone_or_update() {
          URL="$1"
          DIR="$2"

          if [ -d "$DIR/.git" ]; then
            git -C "$DIR" pull --ff-only
          else
            git clone "$URL" "$DIR"
          fi
        }

        clone_or_update \
          https://github.com/enactic/openarm_description.git \
          openarm_description

        clone_or_update \
          https://github.com/enactic/openarm_ros2.git \
          openarm_ros2

        clone_or_update \
          https://github.com/enactic/openarm_can.git \
          openarm_can

        echo "[V19] OpenArm vendor repositories ready."
        ''',
    )

    write(
        "scripts/build_ros2_ws.sh",
        r'''#!/usr/bin/env bash
        set -euo pipefail

        ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
        WS="$ROOT/ros2_ws"

        if [ -z "${ROS_DISTRO:-}" ]; then
          echo "请先 source /opt/ros/<distro>/setup.bash"
          exit 1
        fi

        cd "$WS"

        rosdep install \
          --from-paths src \
          --ignore-src \
          -r \
          -y

        colcon build --symlink-install

        echo
        echo "source $WS/install/setup.bash"
        ''',
    )

    write(
        "scripts/test_ros2_task.sh",
        r'''#!/usr/bin/env bash
        ros2 topic pub --once \
          /openarm/task_command \
          std_msgs/msg/String \
          "{data: '{\"task_id\":\"v19_right_001\",\"position\":[0.516,0.050,1.050]}'}"
        ''',
    )


def create_docs() -> None:
    write(
        "docs/ROADMAP.md",
        """
        # V19 路线

        ## V19.0

        - V18 仿真快照独立保存
        - Arm Interface
        - Windows Simulation Bridge
        - ROS2 dry-run
        - 自动选臂
        - 工作空间安全检查
        - FollowJointTrajectory 接口
        - 官方依赖安装脚本

        ## V19.1

        - openarm_description URDF 对齐
        - RViz 双臂显示
        - joint name 对齐
        - controller name 对齐
        - TF 坐标树验证

        ## V19.2

        - MoveIt2 IK
        - 单臂 home
        - 单关节小角度运动
        - pregrasp 无物体轨迹

        ## V19.3

        - 相机标定
        - camera frame 到 base_link
        - 右臂规则抓取真机测试

        ## V19.4

        - 双臂自动选臂
        - inactive arm park
        - 双臂距离监控
        - 碰撞后停止

        ## V19.5

        - Sim2Real 参数随机化
        - RL 局部策略
        - rule fallback
        """,
    )

    write(
        "docs/REAL_ROBOT_SAFETY.md",
        """
        # 真机首次运行安全要求

        1. 保持 dry_run=true。
        2. 首次只测试一个关节。
        3. 首次目标变化不超过约 2 度。
        4. 速度和加速度比例设为 0.1。
        5. 清空工作区域。
        6. 操作者手放急停。
        7. 不携带物体。
        8. 不部署 RL 策略。
        9. 先右臂，后左臂，最后双臂。
        10. 发现方向、零位或关节顺序异常立即停止。
        """,
    )


def make_zip() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(
        ZIP_PATH,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for path in V19_ROOT.rglob("*"):
            if path.is_file():
                archive.write(
                    path,
                    arcname=(
                        Path(V19_ROOT.name)
                        / path.relative_to(V19_ROOT)
                    ),
                )

    print()
    print("=" * 90)
    print("V19 PACKAGE CREATED")
    print("=" * 90)
    print("Directory:", V19_ROOT)
    print("ZIP:", ZIP_PATH)
    print("=" * 90)


def main() -> None:
    print("V18 root:", V18_ROOT)
    print("V19 root:", V19_ROOT)

    if not (V18_ROOT / "scripts").exists():
        raise SystemExit(
            "当前目录不像 OpenArm 项目根目录：缺少 scripts。"
        )

    if V19_ROOT.exists():
        shutil.rmtree(V19_ROOT)

    V19_ROOT.mkdir(parents=True)

    create_snapshot()
    create_common_files()
    create_arm_interface()
    create_windows_bridge()
    create_ros2_package()
    create_vendor_scripts()
    create_docs()
    make_zip()


if __name__ == "__main__":
    main()
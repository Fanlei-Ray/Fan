from __future__ import annotations

"""Entry point for the OpenArm bimanual task planner V11.

Run from the project root:
    python scripts\task_planner\run_demo.py
    python scripts\task_planner\run_demo.py --queue-demo
    python scripts\task_planner\run_demo.py --no-viewer
"""

from pathlib import Path
import sys

TASK_PLANNER_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = TASK_PLANNER_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from task_planner.core import main


if __name__ == "__main__":
    main()

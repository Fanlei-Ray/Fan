from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


APP_ROOT = Path(__file__).resolve().parent
SRC = APP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenArm course upper-computer MVP")
    parser.add_argument(
        "--config",
        type=Path,
        default=APP_ROOT / "config" / "default.json",
        help="JSON configuration file",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="validate config and replay data without opening a GUI",
    )
    args = parser.parse_args()

    # Prevent Python bytecode artifacts outside this E-drive project tree.
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    from openarm_upper.app import run_application, run_self_check

    if args.self_check:
        return run_self_check(args.config)
    return run_application(args.config)


if __name__ == "__main__":
    raise SystemExit(main())

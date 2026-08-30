from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("configuration root must be an object")

    for section in ("app", "transport", "vision", "workspace"):
        if section not in data or not isinstance(data[section], dict):
            raise ValueError(f"missing configuration section: {section}")

    base = path.parent
    data["_config_path"] = str(path)
    data["_config_dir"] = str(base)
    data["app"]["runtime_dir"] = str(
        (base / data["app"]["runtime_dir"]).resolve()
    )
    data["transport"]["replay_file"] = str(
        (base / data["transport"]["replay_file"]).resolve()
    )
    return data

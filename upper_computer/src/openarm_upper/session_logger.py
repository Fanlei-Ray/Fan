from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any


def _json_ready(value: Any) -> Any:
    """Recursively convert nested application values to JSON-safe data."""
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, bytes):
        return {"bytes_omitted": len(value)}
    return value


class SessionLogger:
    def __init__(self, runtime_dir: Path):
        runtime_dir = Path(runtime_dir)
        runtime_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = runtime_dir / f"upper_computer_session_{stamp}.jsonl"

    def log(self, event: str, payload: Any = None) -> None:
        record = {
            "stamp_ns": time.time_ns(),
            "event": event,
            "payload": _json_ready(payload),
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

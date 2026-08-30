from __future__ import annotations

from pathlib import Path
import tkinter as tk

from .config import load_config
from .safety import SafetyValidator
from .session_logger import SessionLogger
from .transports.replay import ReplayTransport


def _make_transport(config):
    mode = config["transport"]["mode"]
    if mode == "replay":
        return ReplayTransport(
            Path(config["transport"]["replay_file"]),
            loop=bool(config["transport"].get("loop", True)),
        )
    if mode == "legacy_vision_ws":
        from .transports.legacy_vision_ws import LegacyVisionWebSocketTransport

        settings = config["transport"]
        return LegacyVisionWebSocketTransport(
            str(settings["websocket_url"]),
            reconnect_min_sec=float(settings.get("reconnect_min_sec", 0.5)),
            reconnect_max_sec=float(settings.get("reconnect_max_sec", 5.0)),
            max_message_bytes=int(settings.get("max_message_bytes", 4_000_000)),
            queue_size=int(settings.get("queue_size", 3)),
        )
    if mode == "mujoco_vision_ws":
        from .transports.mujoco_vision_ws import MujocoVisionWebSocketTransport

        settings = config["transport"]
        return MujocoVisionWebSocketTransport(
            str(settings["websocket_url"]),
            reconnect_min_sec=float(settings.get("reconnect_min_sec", 0.5)),
            reconnect_max_sec=float(settings.get("reconnect_max_sec", 5.0)),
            max_message_bytes=int(settings.get("max_message_bytes", 4_000_000)),
            queue_size=int(settings.get("queue_size", 3)),
        )
    raise ValueError(f"unsupported transport.mode={mode!r}")


def run_self_check(config_path: Path) -> int:
    config = load_config(config_path)
    transport = _make_transport(config)
    checked = 0
    accepted = 0
    if config["transport"]["mode"] == "replay":
        validator = SafetyValidator(config)
        for batch in transport._items:
            fresh = batch.refreshed()
            for detection in fresh.detections:
                checked += 1
                accepted += int(validator.validate(fresh, detection).accepted)
    runtime = Path(config["app"]["runtime_dir"])
    if not str(runtime).lower().startswith(str(Path(__file__).resolve().parents[2]).lower()):
        raise RuntimeError("runtime_dir must stay inside upper_computer")
    print(
        f"SELF-CHECK OK: mode={config['transport']['mode']}, "
        f"{checked} detections, {accepted} accepted"
    )
    print(f"Config: {config['_config_path']}")
    if config["transport"]["mode"] == "replay":
        print(f"Replay: {config['transport']['replay_file']}")
    else:
        print(f"Vision WebSocket: {config['transport']['websocket_url']}")
    print(f"Runtime: {runtime}")
    return 0


def run_application(config_path: Path) -> int:
    config = load_config(config_path)
    transport = _make_transport(config)
    logger = SessionLogger(Path(config["app"]["runtime_dir"]))
    root = tk.Tk()
    from .ui.main_window import MainWindow

    MainWindow(root, config, transport, logger)
    root.mainloop()
    return 0

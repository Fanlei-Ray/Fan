from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "src"))

from openarm_upper.protocols.usart10 import Usart10StreamParser


def import_serial():
    try:
        import serial
        from serial.tools import list_ports
    except ImportError as exc:
        raise SystemExit(
            "pyserial is not installed. Install requirements-serial.txt into an "
            "E-drive project-local virtual environment before live monitoring."
        ) from exc
    return serial, list_ports


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only USART10 telemetry monitor; this tool never writes to the port"
    )
    parser.add_argument("--list", action="store_true", help="list serial ports and exit")
    parser.add_argument("--port", help="explicit serial port, for example COM5 or /dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--no-log", action="store_true", help="do not save JSONL telemetry")
    args = parser.parse_args()
    serial, list_ports = import_serial()

    if args.list:
        for item in list_ports.comports():
            print(f"{item.device}\t{item.description}\t{item.hwid}")
        return 0
    if not args.port:
        parser.error("--port is required unless --list is used")

    log_stream = None
    if not args.no_log:
        runtime = APP_ROOT / "runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_stream = (runtime / f"usart10_j1_telemetry_{stamp}.jsonl").open(
            "a", encoding="utf-8"
        )
        print(f"Log: {log_stream.name}")

    stream_parser = Usart10StreamParser("telemetry")
    print(f"Opening {args.port} at {args.baudrate} bit/s in READ-ONLY application mode")
    print("No serial write call exists in this tool. Press Ctrl+C to stop.")
    try:
        with serial.Serial(args.port, args.baudrate, timeout=0.05) as port:
            while True:
                chunk = port.read(max(1, port.in_waiting))
                if not chunk:
                    continue
                for frame in stream_parser.feed(chunk):
                    record = {
                        "stamp_ns": time.time_ns(),
                        "source": "DM_Data_t[0]/J1",
                        **frame.to_dict(),
                    }
                    line = json.dumps(record, ensure_ascii=False)
                    print(line)
                    if log_stream:
                        log_stream.write(line + "\n")
                        log_stream.flush()
    except KeyboardInterrupt:
        print("Stopped")
    finally:
        if log_stream:
            log_stream.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "src"))

from openarm_upper.protocols.usart10 import (
    Usart10Frame, Usart10StreamParser, decode_command, decode_telemetry, encode_command,
)


def frame_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state", type=int, required=True)
    parser.add_argument("--p-int", type=int, required=True)
    parser.add_argument("--v-int", type=int, required=True)
    parser.add_argument("--t-int", type=int, required=True)
    parser.add_argument("--position", type=float, required=True)
    parser.add_argument("--velocity", type=float, required=True)
    parser.add_argument("--torque", type=float, required=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Safe offline codec for USART10 protocol")
    subparsers = parser.add_subparsers(dest="command", required=True)
    encode = subparsers.add_parser("encode", help="encode but do not transmit")
    frame_arguments(encode)
    decode = subparsers.add_parser("decode", help="decode exactly one hex frame")
    decode.add_argument("hex_frame")
    decode.add_argument("--direction", choices=("telemetry", "command"), default="telemetry")
    parse_file = subparsers.add_parser("parse-file", help="parse a captured binary stream")
    parse_file.add_argument("path", type=Path)
    parse_file.add_argument("--direction", choices=("telemetry", "command"), default="telemetry")
    args = parser.parse_args()

    if args.command == "encode":
        frame = Usart10Frame(
            args.state, args.p_int, args.v_int, args.t_int,
            args.position, args.velocity, args.torque,
        )
        print(encode_command(frame).hex(" ").upper())
        print("OFFLINE ONLY: frame was not transmitted")
    elif args.command == "decode":
        raw = bytes.fromhex(args.hex_frame)
        frame = decode_telemetry(raw) if args.direction == "telemetry" else decode_command(raw)
        print(json.dumps(frame.to_dict(), ensure_ascii=False, indent=2))
    else:
        stream_parser = Usart10StreamParser(args.direction)
        frames = stream_parser.feed(args.path.read_bytes())
        for index, frame in enumerate(frames):
            print(json.dumps({"index": index, **frame.to_dict()}, ensure_ascii=False))
        print(json.dumps({
            "frames": len(frames), "discarded_bytes": stream_parser.discarded_bytes,
            "invalid_frames": stream_parser.invalid_frames,
            "buffered_bytes": len(stream_parser.buffer),
        }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

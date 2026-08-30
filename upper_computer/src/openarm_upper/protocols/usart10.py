from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import struct
from typing import Any, Literal


FRAME_SIZE = 24
COMMAND_HEADER = b"VG"
COMMAND_TAIL = b"NE"
TELEMETRY_HEADER = b"GV"
TELEMETRY_TAIL = b"EN"
_FRAME_STRUCT = struct.Struct("<2shHHHfff2s")

if _FRAME_STRUCT.size != FRAME_SIZE:
    raise RuntimeError("USART10 frame struct is not 24 bytes")


@dataclass(frozen=True)
class Usart10Frame:
    state: int
    p_int: int
    v_int: int
    t_int: int
    position: float
    velocity: float
    torque: float

    def __post_init__(self) -> None:
        if not -32768 <= int(self.state) <= 32767:
            raise ValueError("state must fit int16")
        for name in ("p_int", "v_int", "t_int"):
            value = int(getattr(self, name))
            if not 0 <= value <= 65535:
                raise ValueError(f"{name} must fit uint16")
        for name in ("position", "velocity", "torque"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _encode(frame: Usart10Frame, header: bytes, tail: bytes) -> bytes:
    return _FRAME_STRUCT.pack(
        header, int(frame.state), int(frame.p_int), int(frame.v_int), int(frame.t_int),
        float(frame.position), float(frame.velocity), float(frame.torque), tail,
    )


def _decode(data: bytes, header: bytes, tail: bytes) -> Usart10Frame:
    if len(data) != FRAME_SIZE:
        raise ValueError(f"USART10 frame must be {FRAME_SIZE} bytes, got {len(data)}")
    values = _FRAME_STRUCT.unpack(data)
    if values[0] != header:
        raise ValueError(f"invalid frame header: {values[0]!r}")
    if values[-1] != tail:
        raise ValueError(f"invalid frame tail: {values[-1]!r}")
    return Usart10Frame(
        state=values[1], p_int=values[2], v_int=values[3], t_int=values[4],
        position=values[5], velocity=values[6], torque=values[7],
    )


def encode_command(frame: Usart10Frame) -> bytes:
    """Encode PC -> STM32 data without transmitting it to hardware."""
    return _encode(frame, COMMAND_HEADER, COMMAND_TAIL)


def decode_command(data: bytes) -> Usart10Frame:
    return _decode(data, COMMAND_HEADER, COMMAND_TAIL)


def encode_telemetry(frame: Usart10Frame) -> bytes:
    """Encode STM32 -> PC telemetry for tests and recorded replay."""
    return _encode(frame, TELEMETRY_HEADER, TELEMETRY_TAIL)


def decode_telemetry(data: bytes) -> Usart10Frame:
    """Decode telemetry, currently sourced only from DM_Data_t[0] / J1."""
    return _decode(data, TELEMETRY_HEADER, TELEMETRY_TAIL)


class Usart10StreamParser:
    """Recover fixed-length frames from fragmented/noisy serial byte streams."""

    def __init__(self, direction: Literal["telemetry", "command"] = "telemetry"):
        if direction == "telemetry":
            self.header, self.tail = TELEMETRY_HEADER, TELEMETRY_TAIL
        elif direction == "command":
            self.header, self.tail = COMMAND_HEADER, COMMAND_TAIL
        else:
            raise ValueError("direction must be 'telemetry' or 'command'")
        self.direction = direction
        self.buffer = bytearray()
        self.discarded_bytes = 0
        self.invalid_frames = 0

    def feed(self, data: bytes | bytearray | memoryview) -> list[Usart10Frame]:
        self.buffer.extend(bytes(data))
        frames: list[Usart10Frame] = []
        while True:
            index = self.buffer.find(self.header)
            if index < 0:
                keep = 1 if self.buffer.endswith(self.header[:1]) else 0
                drop = len(self.buffer) - keep
                self.discarded_bytes += drop
                if drop:
                    del self.buffer[:drop]
                break
            if index:
                self.discarded_bytes += index
                del self.buffer[:index]
            if len(self.buffer) < FRAME_SIZE:
                break
            candidate = bytes(self.buffer[:FRAME_SIZE])
            if candidate[-2:] != self.tail:
                self.invalid_frames += 1
                self.discarded_bytes += 1
                del self.buffer[0]
                continue
            frames.append(_decode(candidate, self.header, self.tail))
            del self.buffer[:FRAME_SIZE]
        return frames

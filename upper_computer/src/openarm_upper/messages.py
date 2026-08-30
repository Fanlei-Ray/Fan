from __future__ import annotations

from dataclasses import asdict, dataclass, field
import base64
import binascii
import math
import time
from typing import Any


def _float_tuple(value: Any, length: int, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{field_name} must contain {length} values")
    result = tuple(float(item) for item in value)
    if not all(item == item and abs(item) != float("inf") for item in result):
        raise ValueError(f"{field_name} contains a non-finite value")
    return result


@dataclass(frozen=True)
class Detection:
    id: str
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    position_m: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    source: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Detection":
        detection_id = str(raw.get("id", "")).strip()
        class_name = str(raw.get("class_name", "")).strip()
        if not detection_id:
            raise ValueError("detection.id is required")
        if not class_name:
            raise ValueError("detection.class_name is required")
        confidence = float(raw.get("confidence", -1.0))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("detection.confidence must be in [0, 1]")
        bbox = _float_tuple(raw.get("bbox_xyxy"), 4, "bbox_xyxy")
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError("bbox_xyxy must have positive width and height")
        return cls(
            id=detection_id,
            class_name=class_name,
            confidence=confidence,
            bbox_xyxy=bbox,
            position_m=_float_tuple(raw.get("position_m"), 3, "position_m"),
            orientation_xyzw=_float_tuple(
                raw.get("orientation_xyzw", [0, 0, 0, 1]),
                4,
                "orientation_xyzw",
            ),
            source=str(raw.get("source", "")).strip(),
        )


@dataclass(frozen=True)
class DetectionBatch:
    schema_version: str
    frame_id: str
    stamp_ns: int
    image_size: tuple[int, int]
    detections: tuple[Detection, ...]
    image_path: str | None = None
    image_jpeg: bytes | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DetectionBatch":
        frame_id = str(raw.get("frame_id", "")).strip()
        if not frame_id:
            raise ValueError("frame_id is required")
        size = raw.get("image_size", [640, 480])
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            raise ValueError("image_size must contain width and height")
        width, height = int(size[0]), int(size[1])
        if width <= 0 or height <= 0:
            raise ValueError("image_size must be positive")
        items = raw.get("detections", [])
        if not isinstance(items, list):
            raise ValueError("detections must be a list")
        image_jpeg = None
        encoded_image = raw.get("image_b64", "")
        if encoded_image:
            if not isinstance(encoded_image, str):
                raise ValueError("image_b64 must be a string")
            if len(encoded_image) > 4_000_000:
                raise ValueError("image_b64 exceeds safety limit")
            try:
                image_jpeg = base64.b64decode(encoded_image, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("image_b64 is not valid base64") from exc
        return cls(
            schema_version=str(raw.get("schema_version", "1.0")),
            frame_id=frame_id,
            stamp_ns=int(raw.get("stamp_ns", 0)),
            image_size=(width, height),
            detections=tuple(Detection.from_dict(item) for item in items),
            image_path=raw.get("image_path"),
            image_jpeg=image_jpeg,
        )

    def refreshed(self, stamp_ns: int | None = None) -> "DetectionBatch":
        return DetectionBatch(
            schema_version=self.schema_version,
            frame_id=self.frame_id,
            stamp_ns=stamp_ns if stamp_ns is not None else time.time_ns(),
            image_size=self.image_size,
            detections=self.detections,
            image_path=self.image_path,
            image_jpeg=self.image_jpeg,
        )


@dataclass(frozen=True)
class LegacyVisionObservation:
    """Observation from the teammate's current WebSocket ``robot_state``.

    ``pixel_center_uv`` is in pixels and ``depth_m`` is only an axial depth.
    This type intentionally cannot be converted to ``Detection`` because the
    payload is not a robot-frame 3D pose and contains no bbox/confidence.
    """

    received_ns: int
    detected: bool
    find_success: bool
    class_name: str
    pixel_center_uv: tuple[int, int]
    depth_m: float
    image_jpeg: bytes | None
    command: str = ""
    command_target: str = ""
    image_size: tuple[int, int] = (640, 480)

    @classmethod
    def from_legacy_dict(
        cls,
        raw: dict[str, Any],
        *,
        max_image_b64_chars: int = 2_000_000,
    ) -> "LegacyVisionObservation":
        if not isinstance(raw, dict):
            raise ValueError("legacy vision message must be a JSON object")
        x, y = int(raw.get("x", 0)), int(raw.get("y", 0))
        depth = float(raw.get("z", 0.0))
        if not math.isfinite(depth) or depth < 0.0:
            raise ValueError("legacy z/depth must be a finite non-negative value")
        image_jpeg = None
        encoded = raw.get("frame_b64", "")
        if encoded:
            if not isinstance(encoded, str):
                raise ValueError("frame_b64 must be a string")
            if len(encoded) > max_image_b64_chars:
                raise ValueError("frame_b64 exceeds configured safety limit")
            try:
                image_jpeg = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("frame_b64 is not valid base64") from exc
        return cls(
            received_ns=time.time_ns(),
            detected=bool(raw.get("detected", False)),
            find_success=bool(raw.get("find_success", False)),
            class_name=str(raw.get("class", "")).strip(),
            pixel_center_uv=(x, y),
            depth_m=depth,
            image_jpeg=image_jpeg,
            command=str(raw.get("command", "")),
            command_target=str(raw.get("command_target", "")),
        )

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "received_ns": self.received_ns,
            "detected": self.detected,
            "find_success": self.find_success,
            "class_name": self.class_name,
            "pixel_center_uv": self.pixel_center_uv,
            "depth_m": self.depth_m,
            "has_image": self.image_jpeg is not None,
            "command": self.command,
            "command_target": self.command_target,
            "coordinate_status": "pixel_plus_depth_only_not_robot_xyz",
        }

@dataclass(frozen=True)
class TaskCommand:
    task_id: str
    command: str
    object_id: str | None
    frame_id: str
    position_m: tuple[float, float, float] | None
    orientation_xyzw: tuple[float, float, float, float] | None
    object_class: str | None = None
    requested_arm: str = "auto"
    dry_run: bool = True
    created_ns: int = field(default_factory=time.time_ns)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskStatus:
    task_id: str
    state: str
    progress: float
    selected_arm: str | None = None
    error_code: str | None = None
    message: str = ""
    stamp_ns: int = field(default_factory=time.time_ns)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MotorHealth:
    connected: bool = False
    enabled: bool = False
    can_state: str = "待接入"
    temperatures_c: tuple[float, ...] = ()
    faults: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

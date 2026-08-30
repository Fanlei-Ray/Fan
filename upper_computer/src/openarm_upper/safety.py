from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from .messages import Detection, DetectionBatch


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    reasons: tuple[str, ...]


class SafetyValidator:
    def __init__(self, config: dict[str, Any]):
        vision = config["vision"]
        workspace = config["workspace"]
        self.expected_frame = str(vision["expected_frame"])
        self.minimum_confidence = float(vision["minimum_confidence"])
        self.maximum_age_sec = float(vision["maximum_age_sec"])
        self.allowed_classes = set(str(x) for x in vision["allowed_classes"])
        self.x_range = tuple(float(x) for x in workspace["x_m"])
        self.y_range = tuple(float(x) for x in workspace["y_m"])
        self.z_range = tuple(float(x) for x in workspace["z_m"])

    def validate(
        self,
        batch: DetectionBatch,
        detection: Detection,
        now_ns: int | None = None,
    ) -> ValidationResult:
        reasons: list[str] = []
        now = now_ns if now_ns is not None else time.time_ns()
        age_sec = (now - batch.stamp_ns) / 1_000_000_000

        if batch.frame_id != self.expected_frame:
            reasons.append(
                f"坐标系必须是 {self.expected_frame}，收到 {batch.frame_id}"
            )
        if batch.stamp_ns <= 0 or age_sec < -0.2 or age_sec > self.maximum_age_sec:
            reasons.append(f"视觉消息过期或时间戳异常（age={age_sec:.2f}s）")
        if detection.class_name not in self.allowed_classes:
            reasons.append(f"类别不允许执行：{detection.class_name}")
        if detection.confidence < self.minimum_confidence:
            reasons.append(
                f"置信度 {detection.confidence:.2f} 低于 {self.minimum_confidence:.2f}"
            )

        x, y, z = detection.position_m
        for name, value, limits in (
            ("x", x, self.x_range),
            ("y", y, self.y_range),
            ("z", z, self.z_range),
        ):
            if not limits[0] <= value <= limits[1]:
                reasons.append(
                    f"{name}={value:.3f}m 超出 [{limits[0]:.3f}, {limits[1]:.3f}]"
                )
        return ValidationResult(not reasons, tuple(reasons))

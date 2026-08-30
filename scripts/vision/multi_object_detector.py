from __future__ import annotations

"""Hybrid YOLOv8 / rendered-RGB detector for the MuJoCo course demo.

Generic COCO weights can recognise realistic phones and mice, but they are not
guaranteed to recognise the simple procedural geometry used by MuJoCo.  In
``auto`` mode this module runs YOLOv8 first and fills missing simulation classes
with deterministic RGB segmentation.  Every result records its true source so
the fallback cannot be presented as a YOLO result.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from vision.vision_cube_detector import CAMERA_NAME, OrangeCubeDetector


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = (
    ROOT / "upper_computer" / "vendor" / "yolov8_reference" / "yolov8n.pt"
)
DEFAULT_SEG_WEIGHTS = (
    ROOT / "upper_computer" / "vendor" / "yolov8_reference" / "yolov8n-seg.pt"
)
YOLO_CONFIG_DIR = ROOT / "upper_computer" / "work" / "ultralytics"


@dataclass(frozen=True)
class SimulationObjectSpec:
    body_name: str
    class_name: str
    display_name: str
    stable_z: float
    planner_spawn_z: float
    mask_kind: str
    yolo_names: tuple[str, ...] = ()
    projection_plane_z: float | None = None
    world_xy_offset: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class ObjectDetection:
    object_id: str
    body_name: str
    class_name: str
    display_name: str
    confidence: float
    bbox: tuple[int, int, int, int]
    pixel_center: tuple[float, float]
    stable_world_pos: np.ndarray
    planner_cube_pos: np.ndarray
    source: str
    image_angle_rad: float = 0.0


DEFAULT_OBJECT_SPECS = (
    SimulationObjectSpec(
        "orange_cube", "orange_cube", "橙色方块", 1.025, 1.050, "orange"
    ),
    SimulationObjectSpec(
        "blue_part", "blue_part", "蓝色圆柱零件", 1.025, 1.050, "cyan"
    ),
    SimulationObjectSpec(
        "phone", "phone", "手机模型", 1.025, 1.050, "magenta", ("cell phone",)
    ),
    SimulationObjectSpec(
        "mouse", "mouse", "鼠标模型", 1.025, 1.050, "green", ("mouse",)
    ),
)


class MultiObjectDetector:
    """Detect all configured objects and back-project their RGB box centres."""

    def __init__(
        self,
        model: Any,
        data: Any,
        *,
        backend: str = "auto",
        weights: str | Path = DEFAULT_WEIGHTS,
        confidence: float = 0.35,
        specs: Iterable[SimulationObjectSpec] = DEFAULT_OBJECT_SPECS,
        camera_name: str = CAMERA_NAME,
    ) -> None:
        self.camera = OrangeCubeDetector(model, data, camera_name=camera_name)
        self.model = model
        self.data = data
        self.width = self.camera.width
        self.height = self.camera.height
        self.backend = str(backend).strip().lower()
        if self.backend not in {"auto", "yolo", "color"}:
            raise ValueError("backend must be one of: auto, yolo, color")
        self.weights = Path(weights)
        self.confidence = float(confidence)
        self.specs = tuple(specs)
        self.spec_by_class = {spec.class_name: spec for spec in self.specs}
        self.yolo_name_to_spec = {
            name: spec for spec in self.specs for name in spec.yolo_names
        }
        self.yolo_model: Any | None = None
        self.yolo_error = "disabled"
        if self.backend in {"auto", "yolo"}:
            self._load_yolo()

    @property
    def status(self) -> str:
        if self.yolo_model is not None:
            if self.backend == "yolo":
                return f"YOLOv8 strict ({self.weights.name}; no RGB fallback)"
            return f"YOLOv8({self.weights.name}) + RGB fallback"
        if self.backend == "color":
            return "rendered RGB multi-color detector"
        return f"RGB fallback; YOLO unavailable: {self.yolo_error}"

    def _load_yolo(self) -> None:
        try:
            YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))
            os.environ.setdefault("TORCH_HOME", str(ROOT / "upper_computer" / "work" / "torch"))
            from ultralytics import YOLO

            if not self.weights.exists():
                raise FileNotFoundError(f"weights not found: {self.weights}")
            self.yolo_model = YOLO(str(self.weights))
            self.yolo_error = ""
        except Exception as exc:
            self.yolo_model = None
            self.yolo_error = f"{type(exc).__name__}: {exc}"
            if self.backend == "yolo":
                raise RuntimeError(f"YOLOv8 requested but unavailable: {self.yolo_error}") from exc

    def render_rgb(self) -> np.ndarray:
        return self.camera.render_rgb()

    @staticmethod
    def color_mask(rgb: np.ndarray, kind: str) -> np.ndarray:
        arr = np.asarray(rgb, dtype=np.float32)
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        if kind == "orange":
            return (
                (r > 140.0)
                & (g > 70.0)
                & (b < 180.0)
                & (r > g * 1.05)
                & (g > b * 1.25)
            )
        if kind == "blue":
            return (b > 105.0) & (b > r * 1.35) & (b > g * 1.15)
        if kind == "cyan":
            return (g > 105.0) & (b > 110.0) & (r < 130.0) & (b > r * 1.35)
        if kind == "magenta":
            return (r > 105.0) & (b > 55.0) & (r > g * 1.45) & (b > g * 1.20)
        if kind == "green":
            return (g > 90.0) & (g > r * 1.30) & (g > b * 1.12)
        raise ValueError(f"unknown mask kind: {kind}")

    @staticmethod
    def _mask_box(mask: np.ndarray, minimum_area: int = 20) -> tuple[int, int, int, int] | None:
        """Return the largest compact component inside the table-camera ROI.

        Some robot decorations and the visible camera marker are blue.  A raw
        min/max over every blue pixel would therefore create a box spanning the
        whole frame. Connected components keep the detector tied to the compact
        tabletop object instead.
        """
        height, width = mask.shape[:2]
        try:
            import cv2

            count, _, stats, centroids = cv2.connectedComponentsWithStats(
                np.asarray(mask, dtype=np.uint8), connectivity=8
            )
            candidates: list[tuple[int, tuple[int, int, int, int]]] = []
            for index in range(1, count):
                x, y, box_w, box_h, area = (int(value) for value in stats[index])
                cx, cy = (float(value) for value in centroids[index])
                if area < minimum_area:
                    continue
                if not (0.12 * width <= cx <= 0.72 * width):
                    continue
                if not (0.18 * height <= cy <= 0.82 * height):
                    continue
                if box_w > 0.22 * width or box_h > 0.30 * height:
                    continue
                candidates.append((area, (x, y, x + box_w - 1, y + box_h - 1)))
            if candidates:
                return max(candidates, key=lambda item: item[0])[1]
            return None
        except Exception:
            ys, xs = np.where(mask)
            if len(xs) < minimum_area:
                return None
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            if x1 - x0 > 0.22 * width or y1 - y0 > 0.30 * height:
                return None
            return x0, y0, x1, y1

    def _make_detection(
        self,
        spec: SimulationObjectSpec,
        bbox: tuple[int, int, int, int],
        confidence: float,
        source: str,
        pixel_center: tuple[float, float] | None = None,
        image_angle_rad: float = 0.0,
    ) -> ObjectDetection:
        x0, y0, x1, y1 = bbox
        if pixel_center is None:
            u = 0.5 * (x0 + x1)
            v = 0.5 * (y0 + y1)
        else:
            u, v = (float(value) for value in pixel_center)
        projection_plane_z = (
            spec.stable_z
            if spec.projection_plane_z is None
            else float(spec.projection_plane_z)
        )
        stable = self.camera.pixel_to_world_on_plane(u, v, plane_z=projection_plane_z)
        stable[:2] += np.asarray(spec.world_xy_offset, dtype=float)
        stable[2] = spec.stable_z
        planner = np.array([stable[0], stable[1], spec.planner_spawn_z], dtype=float)
        return ObjectDetection(
            object_id=f"sim-{spec.body_name}",
            body_name=spec.body_name,
            class_name=spec.class_name,
            display_name=spec.display_name,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            bbox=bbox,
            pixel_center=(u, v),
            stable_world_pos=stable,
            planner_cube_pos=planner,
            source=source,
            image_angle_rad=float(image_angle_rad),
        )

    @staticmethod
    def _mask_geometry(
        polygon: Any,
        fallback_bbox: tuple[int, int, int, int],
    ) -> tuple[tuple[int, int, int, int], tuple[float, float], float]:
        """Return mask bbox, centroid and principal-axis angle in image space."""
        points = np.asarray(polygon, dtype=float).reshape(-1, 2)
        if len(points) < 3:
            x0, y0, x1, y1 = fallback_bbox
            return fallback_bbox, (0.5 * (x0 + x1), 0.5 * (y0 + y1)), 0.0
        x0, y0 = np.floor(points.min(axis=0)).astype(int)
        x1, y1 = np.ceil(points.max(axis=0)).astype(int)
        center = points.mean(axis=0)
        centered = points - center
        covariance = centered.T @ centered / max(1, len(points) - 1)
        values, vectors = np.linalg.eigh(covariance)
        principal = vectors[:, int(np.argmax(values))]
        angle = float(np.arctan2(principal[1], principal[0]))
        return (int(x0), int(y0), int(x1), int(y1)), (float(center[0]), float(center[1])), angle

    def _detect_yolo(self, rgb: np.ndarray) -> list[ObjectDetection]:
        if self.yolo_model is None:
            return []
        # Ultralytics treats numpy images as OpenCV-style BGR. MuJoCo renders
        # RGB, so swap channels here; passing RGB directly made red apples look
        # blue to the network while the UI preview itself looked correct.
        bgr = np.ascontiguousarray(np.asarray(rgb)[..., ::-1])
        results = self.yolo_model.predict(
            source=bgr, conf=self.confidence, verbose=False, device="cpu"
        )
        detections: list[ObjectDetection] = []
        for result in results:
            names = result.names
            boxes = result.boxes
            if boxes is None:
                continue
            mask_polygons = result.masks.xy if result.masks is not None else None
            for index, (xyxy, conf, class_id) in enumerate(
                zip(boxes.xyxy, boxes.conf, boxes.cls)
            ):
                name = str(names[int(class_id)]).strip().lower()
                spec = self.yolo_name_to_spec.get(name)
                if spec is None:
                    continue
                values = [int(round(float(value))) for value in xyxy.tolist()]
                x0, y0, x1, y1 = values
                if x1 <= x0 or y1 <= y0:
                    continue
                bbox = (x0, y0, x1, y1)
                pixel_center = None
                image_angle_rad = 0.0
                source_kind = "yolov8"
                if mask_polygons is not None and index < len(mask_polygons):
                    bbox, pixel_center, image_angle_rad = self._mask_geometry(
                        mask_polygons[index], bbox
                    )
                    source_kind = "yolov8-seg"
                detections.append(
                    self._make_detection(
                        spec,
                        bbox,
                        float(conf),
                        f"{source_kind}:{self.weights.name}:{name}",
                        pixel_center=pixel_center,
                        image_angle_rad=image_angle_rad,
                    )
                )
        # One physical simulation body exists per configured class.  Keep only
        # the strongest candidate if COCO produces duplicate boxes.
        strongest: dict[str, ObjectDetection] = {}
        for detection in detections:
            current = strongest.get(detection.class_name)
            if current is None or detection.confidence > current.confidence:
                strongest[detection.class_name] = detection
        return list(strongest.values())

    def _detect_colors(
        self, rgb: np.ndarray, missing_classes: set[str] | None = None
    ) -> list[ObjectDetection]:
        detections: list[ObjectDetection] = []
        for spec in self.specs:
            if missing_classes is not None and spec.class_name not in missing_classes:
                continue
            bbox = self._mask_box(self.color_mask(rgb, spec.mask_kind))
            if bbox is None:
                continue
            detections.append(
                self._make_detection(
                    spec, bbox, 0.96, f"rgb_fallback:{spec.mask_kind}"
                )
            )
        return detections

    def detect_all(self, rgb: np.ndarray | None = None) -> list[ObjectDetection]:
        rgb = self.render_rgb() if rgb is None else np.asarray(rgb)
        yolo_detections = self._detect_yolo(rgb) if self.backend in {"auto", "yolo"} else []
        if self.backend == "yolo":
            return yolo_detections
        detected_classes = {item.class_name for item in yolo_detections}
        missing = set(self.spec_by_class) - detected_classes
        color_detections = self._detect_colors(rgb, missing_classes=missing)
        return sorted(
            [*yolo_detections, *color_detections], key=lambda item: item.class_name
        )

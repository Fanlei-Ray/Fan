from __future__ import annotations

from pathlib import Path
from io import BytesIO
import tkinter as tk
from typing import Any

from ..messages import DetectionBatch, LegacyVisionObservation


class DetectionCanvas(tk.Canvas):
    def __init__(self, master: Any, width: int = 720, height: int = 520):
        super().__init__(
            master,
            width=width,
            height=height,
            bg="#111827",
            highlightthickness=1,
            highlightbackground="#334155",
        )
        self.view_width = width
        self.view_height = height
        self._photo = None
        self._last_batch: DetectionBatch | None = None
        self._last_raw: LegacyVisionObservation | None = None
        self._mode = "batch"
        self._selected_id: str | None = None
        self.bind("<Configure>", self._on_resize)
        self.draw(None)

    def set_selected_id(self, detection_id: str | None) -> None:
        self._selected_id = detection_id
        if self._mode == "batch":
            self.draw(self._last_batch)

    def _on_resize(self, event: Any) -> None:
        self.view_width = max(1, int(event.width))
        self.view_height = max(1, int(event.height))
        if self._mode == "raw":
            self.draw_raw(self._last_raw)
        else:
            self.draw(self._last_batch)

    def draw(self, batch: DetectionBatch | None) -> None:
        self._mode = "batch"
        self._last_batch = batch
        self._last_raw = None
        self.delete("all")
        source_w, source_h = (640, 480) if batch is None else batch.image_size
        margin = 18
        scale = min(
            (self.view_width - margin * 2) / source_w,
            (self.view_height - margin * 2) / source_h,
        )
        draw_w, draw_h = source_w * scale, source_h * scale
        ox = (self.view_width - draw_w) / 2
        oy = (self.view_height - draw_h) / 2

        image_drawn = False
        if batch and batch.image_jpeg:
            try:
                from PIL import Image, ImageTk

                image = Image.open(BytesIO(batch.image_jpeg)).convert("RGB")
                image = image.resize((int(draw_w), int(draw_h)))
                self._photo = ImageTk.PhotoImage(image)
                self.create_image(ox, oy, image=self._photo, anchor="nw")
                image_drawn = True
            except Exception:
                image_drawn = False
        elif batch and batch.image_path and Path(batch.image_path).exists():
            try:
                from PIL import Image, ImageTk

                image = Image.open(batch.image_path).convert("RGB")
                image = image.resize((int(draw_w), int(draw_h)))
                self._photo = ImageTk.PhotoImage(image)
                self.create_image(ox, oy, image=self._photo, anchor="nw")
                image_drawn = True
            except Exception:
                image_drawn = False

        if not image_drawn:
            self.create_rectangle(ox, oy, ox + draw_w, oy + draw_h, fill="#0f172a", outline="#475569")
            self.create_text(
                ox + draw_w / 2,
                oy + draw_h / 2,
                text="视觉图像待接入\n当前显示检测结果回放",
                fill="#94a3b8",
                font=("Microsoft YaHei UI", 16),
                justify="center",
            )

        if not batch:
            return
        colors = ("#22c55e", "#38bdf8", "#f59e0b", "#e879f9")
        for index, det in enumerate(batch.detections):
            x0, y0, x1, y1 = det.bbox_xyxy
            box = (ox + x0 * scale, oy + y0 * scale, ox + x1 * scale, oy + y1 * scale)
            color = colors[index % len(colors)]
            selected = det.id == self._selected_id
            self.create_rectangle(*box, outline=color, width=5 if selected else 2)
            label = f"{'▶ ' if selected else ''}{det.class_name}  {det.confidence:.2f}"
            self.create_rectangle(box[0], box[1] - 24, box[0] + max(150, len(label) * 9), box[1], fill=color, outline=color)
            self.create_text(box[0] + 6, box[1] - 12, text=label, anchor="w", fill="#020617", font=("Consolas", 10, "bold"))

    def draw_raw(self, observation: LegacyVisionObservation | None) -> None:
        """Draw the legacy pixel+depth stream without implying robot XYZ."""
        self._mode = "raw"
        self._last_raw = observation
        self._last_batch = None
        self.delete("all")
        source_w, source_h = (640, 480) if observation is None else observation.image_size
        margin = 18
        scale = min(
            (self.view_width - margin * 2) / source_w,
            (self.view_height - margin * 2) / source_h,
        )
        draw_w, draw_h = source_w * scale, source_h * scale
        ox = (self.view_width - draw_w) / 2
        oy = (self.view_height - draw_h) / 2

        image_drawn = False
        if observation and observation.image_jpeg:
            try:
                from PIL import Image, ImageTk

                image = Image.open(BytesIO(observation.image_jpeg)).convert("RGB")
                image = image.resize((int(draw_w), int(draw_h)))
                self._photo = ImageTk.PhotoImage(image)
                self.create_image(ox, oy, image=self._photo, anchor="nw")
                image_drawn = True
            except Exception:
                image_drawn = False
        if not image_drawn:
            self.create_rectangle(
                ox, oy, ox + draw_w, oy + draw_h,
                fill="#0f172a", outline="#475569",
            )
            self.create_text(
                ox + draw_w / 2,
                oy + draw_h / 2,
                text="等待视觉同学 WebSocket JPEG 画面",
                fill="#94a3b8",
                font=("Microsoft YaHei UI", 15),
            )

        if observation and observation.detected:
            u, v = observation.pixel_center_uv
            x, y = ox + u * scale, oy + v * scale
            color = "#22c55e" if observation.find_success else "#f59e0b"
            radius = 9
            self.create_oval(x - radius, y - radius, x + radius, y + radius, outline=color, width=3)
            self.create_line(x - 16, y, x + 16, y, fill=color, width=2)
            self.create_line(x, y - 16, x, y + 16, fill=color, width=2)
            label = f"{observation.class_name or 'unknown'}  u={u} v={v} depth={observation.depth_m:.3f}m"
            self.create_rectangle(x + 12, y - 28, x + 390, y - 2, fill=color, outline=color)
            self.create_text(
                x + 18, y - 15, text=label, anchor="w",
                fill="#020617", font=("Consolas", 10, "bold"),
            )

        banner_y = oy + draw_h - 30
        self.create_rectangle(ox, banner_y, ox + draw_w, oy + draw_h, fill="#991b1b", outline="")
        self.create_text(
            ox + draw_w / 2,
            banner_y + 15,
            text="PIXEL + DEPTH ONLY · 未标定 · 禁止机械臂执行",
            fill="white",
            font=("Microsoft YaHei UI", 11, "bold"),
        )

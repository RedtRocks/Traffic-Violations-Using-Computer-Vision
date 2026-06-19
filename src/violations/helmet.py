"""
Helmet non-compliance detector.
Crops the head ROI from the top fraction of a motorcycle bounding box,
runs a pretrained YOLOv8 helmet classifier, and returns a violation if
no helmet is detected.
"""

import numpy as np
from datetime import datetime
from ultralytics import YOLO
from src.models import TrackedObject, ViolationRecord
from src.violations.classifier import route


class HelmetChecker:
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.55,
        device: str = "cpu",
        head_roi_fraction: float = 0.35,
    ):
        self.model = YOLO(model_path)
        self.conf = conf_threshold
        self.device = device
        self.head_roi_fraction = head_roi_fraction

    def check(
        self,
        tracks: list[TrackedObject],
        frame: np.ndarray,
        frame_id: int,
        camera_id: str = "cam_001",
    ) -> list[ViolationRecord]:
        violations: list[ViolationRecord] = []
        motorcycles = [t for t in tracks if t.class_name == "motorcycle"]

        for moto in motorcycles:
            crop = self._crop_head(frame, moto.bbox)
            if crop is None:
                continue

            results = self.model.predict(source=crop, conf=self.conf, device=self.device, verbose=False)
            if not results or not results[0].boxes:
                continue

            for box in results[0].boxes:
                label = self.model.names[int(box.cls[0])].lower()
                confidence = float(box.conf[0])
                if "no" in label or "without" in label or label == "no_helmet":
                    record = ViolationRecord(
                        violation_type="helmet",
                        confidence=confidence,
                        vehicle_id=moto.track_id,
                        bbox=moto.bbox,
                        timestamp=datetime.utcnow().isoformat(),
                        frame_id=frame_id,
                        camera_id=camera_id,
                    )
                    violations.append(route(record))
                    break
        return violations

    def _crop_head(self, frame: np.ndarray, bbox: tuple) -> np.ndarray | None:
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        head_y2 = y1 + int(h * self.head_roi_fraction)
        crop = frame[y1:head_y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None
        return crop

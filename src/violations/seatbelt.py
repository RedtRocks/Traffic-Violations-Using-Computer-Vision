"""
Seatbelt non-compliance detector.
Crops the windshield ROI from a car bounding box and runs a lightweight
binary CNN classifier (seatbelt / no_seatbelt).

If the crop is too small or the model is not loaded, the record is
marked "indeterminate" and routed to the human review queue.
"""

import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from src.models import TrackedObject, ViolationRecord
from src.violations.classifier import route


class SeatbeltChecker:
    def __init__(
        self,
        model_path: str | None,
        conf_threshold: float = 0.55,
        device: str = "cpu",
        windshield_top_fraction: float = 0.15,
        windshield_bottom_fraction: float = 0.55,
        min_crop_width: int = 60,
        min_crop_height: int = 40,
    ):
        self.conf = conf_threshold
        self.device = device
        self.top_frac = windshield_top_fraction
        self.bot_frac = windshield_bottom_fraction
        self.min_w = min_crop_width
        self.min_h = min_crop_height
        self.model = self._load_model(model_path) if model_path else None
        self._transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def check(
        self,
        tracks: list[TrackedObject],
        frame: np.ndarray,
        frame_id: int,
        camera_id: str = "cam_001",
    ) -> list[ViolationRecord]:
        violations: list[ViolationRecord] = []
        cars = [t for t in tracks if t.class_name == "car"]

        for car in cars:
            crop = self._crop_windshield(frame, car.bbox)
            if crop is None:
                record = self._indeterminate(car, frame_id, camera_id)
                violations.append(record)
                continue

            if self.model is None:
                record = self._indeterminate(car, frame_id, camera_id)
                violations.append(record)
                continue

            confidence, label = self._classify(crop)
            if label == "no_seatbelt":
                record = ViolationRecord(
                    violation_type="seatbelt",
                    confidence=confidence,
                    vehicle_id=car.track_id,
                    bbox=car.bbox,
                    timestamp=datetime.utcnow().isoformat(),
                    frame_id=frame_id,
                    camera_id=camera_id,
                )
                violations.append(route(record))
        return violations

    def _crop_windshield(self, frame: np.ndarray, bbox: tuple) -> np.ndarray | None:
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        cy1 = y1 + int(h * self.top_frac)
        cy2 = y1 + int(h * self.bot_frac)
        crop = frame[cy1:cy2, x1:x2]
        if crop.shape[0] < self.min_h or crop.shape[1] < self.min_w:
            return None
        return crop

    def _classify(self, crop: np.ndarray) -> tuple[float, str]:
        pil_img = Image.fromarray(crop[:, :, ::-1])
        tensor = self._transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            prob = torch.sigmoid(logits).item()
        if prob >= self.conf:
            return prob, "no_seatbelt"
        return 1.0 - prob, "seatbelt"

    def _indeterminate(self, car: TrackedObject, frame_id: int, camera_id: str) -> ViolationRecord:
        record = ViolationRecord(
            violation_type="seatbelt",
            confidence=0.0,
            vehicle_id=car.track_id,
            bbox=car.bbox,
            timestamp=datetime.utcnow().isoformat(),
            frame_id=frame_id,
            status="indeterminate",
            camera_id=camera_id,
        )
        return record

    def _load_model(self, path: str) -> nn.Module | None:
        try:
            model = _SeatbeltCNN()
            model.load_state_dict(torch.load(path, map_location=self.device))
            model.eval()
            return model.to(self.device)
        except Exception:
            return None


class _SeatbeltCNN(nn.Module):
    """Lightweight binary CNN for windshield crop classification."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Linear(64 * 4 * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

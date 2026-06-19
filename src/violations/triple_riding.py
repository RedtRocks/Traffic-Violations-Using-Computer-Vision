"""
Triple riding violation detector.
Rule: count persons whose centroid overlaps a motorcycle bounding box.
If count >= 3, flag the motorcycle track as a triple-riding violation.
"""

import numpy as np
from datetime import datetime
from src.models import TrackedObject, ViolationRecord
from src.violations.classifier import route


def check(
    tracks: list[TrackedObject],
    frame_id: int,
    camera_id: str = "cam_001",
    min_overlap_iou: float = 0.20,
) -> list[ViolationRecord]:
    violations: list[ViolationRecord] = []

    motorcycles = [t for t in tracks if t.class_name == "motorcycle"]
    persons = [t for t in tracks if t.class_name == "person"]

    for moto in motorcycles:
        riders = [p for p in persons if _overlap_iou(moto.bbox, p.bbox) >= min_overlap_iou]
        if len(riders) >= 3:
            record = ViolationRecord(
                violation_type="triple_riding",
                confidence=1.0,
                vehicle_id=moto.track_id,
                bbox=moto.bbox,
                timestamp=datetime.utcnow().isoformat(),
                frame_id=frame_id,
                camera_id=camera_id,
            )
            violations.append(route(record))
    return violations


def _overlap_iou(box_a: tuple, box_b: tuple) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)

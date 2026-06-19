"""
Multi-object tracker wrapping Ultralytics ByteTrack.
Maintains per-track centroid history for use by rule-based violation modules.
"""

import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
from src.models import Detection, TrackedObject


_HISTORY_LEN = 60   # frames of centroid history to keep per track


class Tracker:
    def __init__(self, track_thresh: float = 0.50, track_buffer: int = 30, match_thresh: float = 0.80):
        self._track_thresh = track_thresh
        self._track_buffer = track_buffer
        self._match_thresh = match_thresh
        # centroid_history[track_id] = deque of (cx, cy) tuples
        self._centroid_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=_HISTORY_LEN))
        # last_seen class per track id
        self._class_map: dict[int, str] = {}

    def update(self, detections: list[Detection], frame: np.ndarray, frame_id: int) -> list[TrackedObject]:
        """
        Run ByteTrack on the current frame's detections.
        Returns a list of TrackedObject with stable IDs and centroid history.
        """
        if not detections:
            return []

        boxes_xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
        scores = np.array([d.confidence for d in detections], dtype=np.float32)
        class_names = [d.class_name for d in detections]

        # Build ultralytics-compatible results manually using ByteTrack internals
        # via the model-free track call.
        # Ultralytics exposes ByteTrack through YOLO.track(); here we call the
        # tracker directly to avoid needing a model object.
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker
        except ImportError:
            from ultralytics.trackers import BYTETracker

        if not hasattr(self, "_byte_tracker"):
            args = type("Args", (), {
                "track_thresh": self._track_thresh,
                "track_buffer": self._track_buffer,
                "match_thresh": self._match_thresh,
                "mot20": False,
            })()
            self._byte_tracker = BYTETracker(args, frame_rate=10)

        h, w = frame.shape[:2]
        # ByteTracker expects [x1, y1, x2, y2, score]
        dets = np.column_stack([boxes_xyxy, scores])
        tracks = self._byte_tracker.update(dets, [h, w], [h, w])

        tracked: list[TrackedObject] = []
        for t in tracks:
            tid = int(t.track_id)
            x1, y1, x2, y2 = map(int, t.tlbr)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            self._centroid_history[tid].append((cx, cy))

            # Match back to a detection to get class name
            matched_class = _match_class(boxes_xyxy, (x1, y1, x2, y2), class_names)
            if matched_class:
                self._class_map[tid] = matched_class
            class_name = self._class_map.get(tid, "unknown")

            tracked.append(TrackedObject(
                track_id=tid,
                class_name=class_name,
                bbox=(x1, y1, x2, y2),
                confidence=float(t.score),
                frame_id=frame_id,
                centroid_history=list(self._centroid_history[tid]),
            ))
        return tracked

    def get_history(self, track_id: int) -> list[tuple[int, int]]:
        return list(self._centroid_history.get(track_id, []))


def _match_class(
    boxes: np.ndarray,
    tracked_bbox: tuple[int, int, int, int],
    class_names: list[str],
) -> str | None:
    if len(boxes) == 0:
        return None
    tx1, ty1, tx2, ty2 = tracked_bbox
    ious = _batch_iou(boxes, np.array([[tx1, ty1, tx2, ty2]], dtype=np.float32))
    best_idx = int(np.argmax(ious))
    if ious[best_idx] > 0.3:
        return class_names[best_idx]
    return None


def _batch_iou(boxes: np.ndarray, query: np.ndarray) -> np.ndarray:
    qx1, qy1, qx2, qy2 = query[0]
    ix1 = np.maximum(boxes[:, 0], qx1)
    iy1 = np.maximum(boxes[:, 1], qy1)
    ix2 = np.minimum(boxes[:, 2], qx2)
    iy2 = np.minimum(boxes[:, 3], qy2)
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area_query = (qx2 - qx1) * (qy2 - qy1)
    union = area_boxes + area_query - inter
    return inter / (union + 1e-6)

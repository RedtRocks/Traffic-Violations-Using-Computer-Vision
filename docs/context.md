# Project Context — Traffic Violation Detection System

> **Purpose of this file**: Paste the contents into any new AI chat session to restore full project context instantly.

---

## What This Project Is

A real-world deployable prototype for **Automated Photo Identification and Classification of Traffic Violations using Computer Vision**.

- End-to-end pipeline: video/camera input → detection → tracking → violation checks → OCR → evidence → database → dashboard
- Built to be demonstrated, not just researched
- Stack: Python, YOLOv8 (Ultralytics), ByteTrack, PaddleOCR, OpenCV, SQLite, Streamlit

---

## Violations in Scope (All 7 — Nothing Dropped)

| Violation | Detection Method |
|-----------|-----------------|
| Helmet non-compliance | YOLOv8 classifier on head crop of motorcycle bbox |
| Seatbelt non-compliance | Binary CNN on windshield crop of car bbox; marks `indeterminate` if crop too small or model not trained |
| Triple riding | Rule: count persons overlapping motorcycle bbox (IoU ≥ 0.20); if ≥ 3 → violation |
| Wrong-side driving | Rule: centroid direction vector vs `allowed_direction_deg` in camera config for N consecutive frames |
| Stop-line violation | Rule: vehicle centroid past virtual line when signal is red/unknown |
| Red-light violation | Rule: confirmed red signal (HSV) + moving vehicle past stop line |
| Illegal parking | Rule: vehicle centroid inside no-parking polygon + stationary for ≥ 3 minutes (configurable) |

**Excluded (not in problem statement):** mobile phone detection, lane violation.

---

## Full System Architecture

```
Video/Image Input
      ↓
Image Preprocessing  (CLAHE low-light, blur detection via Laplacian variance, rain filter)
      ↓
Vehicle & Road User Detection  (YOLOv8m — car, truck, bus, motorcycle, auto-rickshaw, person)
      ↓
ByteTrack  (stable vehicle IDs + centroid history across frames)
      ↓
Violation Detection Engine
  ├── Helmet         → YOLOv8 classifier
  ├── Seatbelt       → Binary CNN (indeterminate fallback)
  ├── Triple Riding  → Person-count rule
  ├── Wrong-side     → Direction vector rule
  ├── Stop-line      → Virtual line + signal state
  ├── Red-light      → Signal ROI (HSV) + vehicle position
  └── Illegal Parking→ Polygon containment + dwell timer
      ↓
Violation Classifier & Confidence Scorer
  (confidence ≥ threshold → auto_flagged | below → review | crop unusable → indeterminate)
      ↓
License Plate Detection (YOLOv8 fine-tuned on Indian plates) + PaddleOCR
      ↓
Evidence Generator  (annotated JPEG + JSON sidecar per violation)
      ↓
SQLite Database  (via SQLAlchemy — upgradeable to Postgres by changing connection string)
      ↓
Streamlit Dashboard  (KPIs, charts by type/date, searchable table, image viewer, CSV export)
```

---

## Project Structure (All Files Created)

```
traffic project/                          ← root (c:\Users\rauna\Videos\traffic project\)
│
├── app.py                                ← CLI entry point
├── requirements.txt
├── README.md
├── .gitignore                            ← comprehensive (covers .pt, .mp4, DB, logs, cache)
│
├── configs/
│   ├── pipeline.yaml                     ← model paths, FPS, device, JPEG quality
│   ├── cameras.yaml                      ← per-camera: stop line, signal ROI, no-parking polygons
│   └── violations.yaml                   ← per-violation: confidence thresholds, dwell times, ROI fractions
│
├── src/
│   ├── config.py                         ← YAML loader (load_pipeline, load_cameras, load_violations)
│   ├── models.py                         ← shared dataclasses: Detection, TrackedObject, ViolationRecord
│   ├── preprocessing/
│   │   └── frame_processor.py            ← process_frame() → (frame, FrameQuality)
│   ├── detection/
│   │   ├── vehicle_detector.py           ← VehicleDetector class (YOLOv8 wrapper)
│   │   └── plate_detector.py             ← PlateDetector class (with crop-coord translation)
│   ├── tracking/
│   │   └── tracker.py                    ← Tracker class (ByteTrack + centroid history deque)
│   ├── violations/
│   │   ├── classifier.py                 ← route(record) → sets status based on confidence threshold
│   │   ├── signal_utils.py               ← detect_signal_state() → "red" | "green" | "unknown"
│   │   ├── triple_riding.py              ← check(tracks, ...) → [ViolationRecord]
│   │   ├── wrong_side.py                 ← check(tracks, ...) → [ViolationRecord]
│   │   ├── stop_line.py                  ← check(tracks, frame, ...) → [ViolationRecord]
│   │   ├── red_light.py                  ← check(tracks, frame, ...) → [ViolationRecord]
│   │   ├── parking.py                    ← check(tracks, ...) → [ViolationRecord]
│   │   ├── helmet.py                     ← HelmetChecker class
│   │   └── seatbelt.py                   ← SeatbeltChecker class + _SeatbeltCNN architecture
│   ├── ocr/
│   │   └── plate_reader.py               ← PlateReader class (PaddleOCR) → PlateReadResult
│   ├── evidence/
│   │   └── generator.py                  ← EvidenceGenerator.save() → annotated JPEG + JSON
│   ├── database/
│   │   ├── schema.py                     ← SQLAlchemy table + get_engine() + init_db()
│   │   └── repository.py                 ← insert_violation, query_violations, count_by_type, export_csv
│   ├── analytics/
│   │   └── stats.py                      ← violation_summary, recent_violations, search
│   └── evaluation/
│       └── metrics.py                    ← classification_metrics, ocr_accuracy, average_precision, FPSTimer
│
├── pipelines/
│   └── video_pipeline.py                 ← main run() — wires all modules, frame loop
│
├── dashboard/
│   └── app.py                            ← Streamlit: KPIs, bar chart, trend line, table, image viewer, CSV
│
├── scripts/
│   ├── download_models.py                ← downloads YOLOv8 base weights via ultralytics
│   └── draw_zones.py                     ← interactive OpenCV tool to click-define stop lines, polygons, ROIs
│
├── models/
│   ├── weights/                          ← .pt files here (gitignored — place manually)
│   └── seatbelt_classifier/              ← seatbelt_cnn.pth (gitignored — train via notebook)
│
├── data/
│   ├── raw/                              ← raw video input (gitignored)
│   ├── samples/                          ← short test clips (not gitignored)
│   └── seatbelt_crops/
│       ├── seatbelt/                     ← labelled windshield crops (worn)
│       └── no_seatbelt/                  ← labelled windshield crops (not worn)
│
├── artifacts/
│   └── evidence/                         ← saved annotated JPEGs + JSON (runtime, gitignored)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_seatbelt_training.ipynb        ← trains _SeatbeltCNN binary classifier
│   └── 03_evaluation.ipynb              ← runs mAP, F1, OCR accuracy, FPS metrics
│
├── tests/
│   ├── test_preprocessing.py             ← CLAHE, blur, rain filter tests
│   ├── test_violations.py                ← triple riding, wrong-side, classifier routing tests
│   └── test_ocr.py                       ← OCR accuracy, IoU, mAP tests
│
└── docs/
    ├── Traffic_Violation_Final_Implementation_Plan.md  ← PRIMARY source of truth
    └── context.md                                      ← this file
```

---

## Key Data Structures

```python
# src/models.py

@dataclass
class Detection:
    class_name: str          # "car", "motorcycle", "person", etc.
    confidence: float
    bbox: tuple              # (x1, y1, x2, y2)
    frame_id: int

@dataclass
class TrackedObject:
    track_id: int
    class_name: str
    bbox: tuple
    confidence: float
    frame_id: int
    centroid_history: list[tuple[int, int]]   # last 60 frames of (cx, cy)

@dataclass
class ViolationRecord:
    violation_type: str      # "helmet", "seatbelt", "triple_riding", etc.
    confidence: float
    vehicle_id: int
    bbox: tuple
    timestamp: str           # ISO format
    frame_id: int
    plate_number: str | None
    plate_confidence: float | None
    status: str              # "auto_flagged" | "review" | "indeterminate"
    evidence_image_path: str | None
    evidence_json_path: str | None
    is_blurry: bool
    camera_id: str
```

---

## Configuration Summary

### `configs/pipeline.yaml`
- `input.source` — video file path / RTSP URL / webcam index
- `input.target_fps` — frames to process per second (default: 10)
- `models.*` — paths to all `.pt` and `.pth` weight files
- `inference.device` — `"cpu"` or `"cuda"`
- `inference.vehicle_conf` / `plate_conf` / `helmet_conf` / `seatbelt_conf`
- `preprocessing.*` — CLAHE clip limit, blur threshold, rain filter toggle
- `evidence.*` — save directory, JPEG quality

### `configs/cameras.yaml`
- Per-camera: `allowed_direction_deg`, `direction_tolerance_deg`
- `stop_line` — two `[x, y]` pixel points
- `signal_roi` — `[x1, y1, x2, y2]` pixel rect
- `no_parking_zones` — list of named polygons (list of `[x, y]` points)
- **Set these up using**: `python scripts/draw_zones.py --video <clip>`

### `configs/violations.yaml`
- Per-violation: `auto_approve_confidence` threshold
- Helmet: `head_roi_fraction`
- Seatbelt: windshield crop fractions, `min_crop_width/height`
- Triple riding: `min_person_overlap_iou`
- Wrong-side: `min_track_frames`, `consecutive_wrong_frames`
- Parking: `dwell_time_seconds`, `stationary_pixel_threshold`
- Tracker: `track_thresh`, `track_buffer`, `match_thresh`

---

## How to Run

```bash
# Full pipeline on a video file
python app.py --video data/samples/test_video.mp4 --camera cam_001 --show

# Webcam
python app.py --video 0

# RTSP stream
python app.py --video rtsp://192.168.1.10/stream

# Preprocessing-only dry run
python app.py --video data/samples/test_video.mp4 --dry-run

# Launch analytics dashboard
python app.py --dashboard

# Run tests
pytest tests/ -v
```

---

## Setup Sequence (First Time)

```bash
pip install -r requirements.txt
python scripts/download_models.py
# Place fine-tuned helmet + plate .pt files → models/weights/
python scripts/draw_zones.py --video data/samples/test_video.mp4
# Paste printed YAML into configs/cameras.yaml
```

---

## What Still Needs Manual Work

| Item | Status | Action Required |
|------|--------|-----------------|
| Helmet model weights | Not included | Fine-tune YOLOv8 on public helmet dataset → `models/weights/helmet_yolov8.pt` |
| Indian plate detector weights | Not included | Fine-tune YOLOv8 on Indian plate dataset → `models/weights/plate_yolov8.pt` |
| Seatbelt CNN weights | Not trained | Collect ~300 windshield crops → run `notebooks/02_seatbelt_training.ipynb` |
| Camera zones | Placeholder values | Run `scripts/draw_zones.py` on actual footage → update `configs/cameras.yaml` |
| Auto-rickshaw class | Uses motorcycle as proxy | Fine-tune YOLOv8 on IDD/DriveIndia dataset if accuracy is poor |
| Evaluation ground truth | None collected | Label 100–200 frames manually → run `notebooks/03_evaluation.ipynb` |

---

## Known Limitations (Documented Honestly)

- **Seatbelt**: Always `indeterminate` until binary CNN is trained (correct behavior — not a bug)
- **Auto-rickshaw**: COCO pretrained model doesn't have this class; `motorcycle` used as proxy
- **Rain handling**: Classical filter only (median blur + unsharp mask); not a full deraining network
- **Signal detection**: HSV-based; may fail under extreme overexposure or nighttime wash-out
- **Multiple cameras**: Supported via `--camera <id>` flag; concurrent streams = multiple processes
- **Parking dwell timer**: Resets on ByteTrack ID switch under occlusion (known tracker limitation)

---

## Tech Stack Summary

| Component | Technology |
|-----------|-----------|
| Vehicle detection | YOLOv8 (Ultralytics) — pretrained COCO |
| Plate detection | YOLOv8 — fine-tuned on Indian plates |
| Helmet classification | YOLOv8 classifier — fine-tuned |
| Seatbelt classification | Custom `_SeatbeltCNN` (PyTorch) — 3-layer CNN |
| Tracking | ByteTrack (via Ultralytics) |
| OCR | PaddleOCR |
| Preprocessing | OpenCV (CLAHE, Laplacian, median blur) |
| Database | SQLite via SQLAlchemy (Postgres-ready) |
| Dashboard | Streamlit |
| Evaluation | scikit-learn + custom mAP |
| Tests | pytest |

---

## Performance Evaluation Targets (per §4.8 of implementation plan)

| Stage | Metric |
|-------|--------|
| Vehicle / helmet / plate detection | mAP@0.5, Precision, Recall |
| Violation classification | Accuracy, F1-score per violation type |
| OCR | Plate-level exact-match accuracy |
| System throughput | FPS on target hardware |
| Scalability | Behaviour under multiple concurrent streams |

---

## Files That Are Safe to Commit to GitHub

Everything **except** what `.gitignore` excludes:
- All `src/`, `pipelines/`, `dashboard/`, `tests/`, `scripts/` Python files ✅
- All `configs/*.yaml` files ✅
- `app.py`, `requirements.txt`, `README.md`, `docs/*.md` ✅
- Empty placeholder dirs (`data/samples/`, `artifacts/`, `models/`) ✅
- **NOT**: `.pt` / `.pth` / `.onnx` weights, `.mp4` videos, SQLite `.db`, evidence images, logs

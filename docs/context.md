# Project Context — Traffic Violation Detection System

> **Purpose of this file**: Paste the contents into any new AI chat session to restore full project context instantly. Kept current as the project evolves.

---

## What This Project Is

A real-world deployable prototype for **Automated Photo Identification and Classification of Traffic Violations using Computer Vision** (AI/ML competition project, must be demonstrable + deployable).

- End-to-end pipeline: video/camera input → preprocessing → detection → tracking → violation checks → OCR → evidence → database → dashboard
- Built to be demonstrated, not just researched
- Stack: Python 3.11, YOLO11 (Ultralytics), self-contained IoU tracker, EasyOCR, OpenCV, SQLite/SQLAlchemy, Streamlit, Docker

---

## Current Status (2026-06-21)

**Working end-to-end on GPU.** Verified: full pipeline runs on a real traffic video on a GTX 1650, generates annotated evidence (illegal_parking + stop_line seen firing), 23/23 unit tests pass.

| Thing | State |
|---|---|
| Environment | Python **3.11.9** venv at `venv/` (system Python is 3.13 which CANNOT install torch/paddle — must use 3.11). torch 2.5.1+cu121, CUDA works on GTX 1650 |
| Vehicle/person model | `models/weights/yolo11s.pt` — COCO pretrained, detects car/bus/truck/motorcycle/person ✅ |
| Helmet model | `models/weights/helmet_yolov8.pt` — TRAINED on Colab. Classes: `rider`, `rider_full_face`, `rider_half_face`, `rider_helmet_invalid`, `rider_no_helmet` ✅ |
| Plate model | `models/weights/plate_yolov8.pt` — TRAINED on Colab. Class: `License_Plate` ✅ |
| OCR | EasyOCR (GPU) — reads plates ~0.90 conf ✅ |
| Seatbelt CNN | Not trained → correctly returns `indeterminate` |
| Git | branch `feature/training-deploy-pipeline` pushed to origin (`github.com/raunaqmittal/...`) |

**Helmet model metrics (best epoch 54, for the report):** overall mAP@50 = 0.578, P = 0.61, R = 0.51. Key violation class **`rider_no_helmet`: mAP@50 = 0.88, P = 0.83, R = 0.80** (strong). Low overall is dragged by a broken 1-sample `rider` class — not worth retraining. **Plate model metrics still to be captured.**

**Open items:** capture plate-model mAP; get a real Indian traffic video with riders/plates for a fuller demo (or use `scripts/images_to_video.py` on the helmet dataset images); optionally build `03_evaluation.ipynb`.

---

## Violations in Scope (All 7 — Nothing Dropped)

| Violation | Detection Method |
|-----------|-----------------|
| Helmet non-compliance | Helmet YOLO run on the **full frame**; `rider_no_helmet` heads associated to motorcycle tracks (upward-expanded-box containment). Not a head-crop classifier — COCO motorcycle boxes exclude the head. |
| Seatbelt non-compliance | Binary CNN on windshield crop of car bbox; marks `indeterminate` if crop too small or model not trained |
| Triple riding | Rule: count persons whose box is ≥ 50% **contained** in the motorcycle box (`min_person_overlap_ratio`); if ≥ 3 → violation. (Containment, not IoU — small rider boxes score low IoU.) |
| Wrong-side driving | Rule: centroid direction vector vs `allowed_direction_deg` in camera config for N consecutive frames |
| Stop-line violation | Rule: vehicle centroid past virtual line when signal is red/unknown |
| Red-light violation | Rule: confirmed red signal (HSV) + moving vehicle past stop line |
| Illegal parking | Rule: vehicle centroid inside no-parking polygon + stationary for ≥ 3 minutes (configurable). Emits `violation_type="illegal_parking"`. |

**Excluded (not in problem statement):** mobile phone detection, lane violation.

---

## Full System Architecture

```
Video/Image Input
      ↓
Image Preprocessing  (CLAHE low-light, blur detection via Laplacian variance, rain filter)
      ↓
Vehicle & Road User Detection  (YOLO11s, COCO — car, truck, bus, motorcycle, person)
      ↓
IoU Tracker  (self-contained greedy IoU association — stable IDs + centroid history)
      ↓
Violation Detection Engine
  ├── Helmet         → full-frame helmet YOLO + associate no-helmet heads to motorcycles
  ├── Seatbelt       → Binary CNN (indeterminate fallback)
  ├── Triple Riding  → person-containment rule
  ├── Wrong-side     → direction vector rule
  ├── Stop-line      → virtual line + signal state
  ├── Red-light      → signal ROI (HSV) + vehicle position
  └── Illegal Parking→ polygon containment + dwell timer
      ↓
Violation Classifier & Confidence Scorer
  (confidence ≥ threshold → auto_flagged | below → review | crop unusable → indeterminate)
      ↓
License Plate Detection (YOLO fine-tuned, class License_Plate) + EasyOCR
      ↓
Evidence Generator  (annotated JPEG + JSON sidecar per violation)
      ↓
SQLite Database  (via SQLAlchemy — upgradeable to Postgres by changing connection string)
      ↓
Streamlit Dashboard  (KPIs, charts by type/date, searchable table, image viewer, CSV export)
```

---

## Project Structure

```
Traffic-Violations-Using-Computer-Vision/
│
├── app.py                                ← CLI entry point
├── requirements.txt                      ← torch+cu121 installed separately; easyocr (not paddle)
├── README.md
├── .gitignore                            ← covers .pt, .mp4, DB+journal, venv, artifacts, cache
├── Dockerfile                            ← Python 3.11 image (CPU; GPU note inside)
├── docker-compose.yml                    ← dashboard + pipeline services, shared volumes
├── .dockerignore
│
├── configs/
│   ├── pipeline.yaml                     ← model paths, FPS, device (cuda), JPEG quality
│   ├── cameras.yaml                      ← per-camera: stop line, signal ROI, no-parking polygons
│   └── violations.yaml                   ← per-violation thresholds, dwell times, ROI fractions
│
├── src/
│   ├── config.py                         ← YAML loaders
│   ├── models.py                         ← dataclasses: Detection, TrackedObject, ViolationRecord
│   ├── preprocessing/frame_processor.py  ← process_frame() → (frame, FrameQuality)
│   ├── detection/
│   │   ├── vehicle_detector.py           ← VehicleDetector (YOLO wrapper, filters to vehicle classes)
│   │   └── plate_detector.py             ← PlateDetector (+ crop-coord translation)
│   ├── tracking/tracker.py               ← Tracker — self-contained IoU tracker (NOT ByteTrack)
│   ├── violations/
│   │   ├── classifier.py                 ← route(record) → status from confidence threshold
│   │   ├── signal_utils.py               ← detect_signal_state() → "red"|"green"|"unknown"
│   │   ├── triple_riding.py / wrong_side.py / stop_line.py / red_light.py / parking.py
│   │   ├── helmet.py                     ← HelmetChecker (full-frame detect + associate)
│   │   └── seatbelt.py                   ← SeatbeltChecker + _SeatbeltCNN
│   ├── ocr/plate_reader.py               ← PlateReader (EasyOCR) → PlateReadResult
│   ├── evidence/generator.py             ← EvidenceGenerator.save() → annotated JPEG + JSON
│   ├── database/{schema.py, repository.py}
│   ├── analytics/stats.py
│   └── evaluation/metrics.py             ← classification_metrics, ocr_accuracy, average_precision, FPSTimer
│
├── pipelines/video_pipeline.py           ← main run() — wires all modules; lazy ML imports for --dry-run
├── dashboard/app.py                      ← Streamlit dashboard
│
├── scripts/
│   ├── download_models.py                ← downloads COCO YOLO vehicle weights
│   ├── download_datasets.py              ← pulls helmet/plate datasets (Roboflow/Kaggle)
│   ├── images_to_video.py                ← build a demo video from an image folder
│   └── draw_zones.py                     ← interactive OpenCV zone editor
│
├── models/weights/                       ← yolo11s.pt, helmet_yolov8.pt, plate_yolov8.pt (gitignored)
├── data/samples/                         ← test clips (gitignored: *.mp4)
├── artifacts/                            ← evidence/ + violations.db (gitignored, runtime)
│
├── notebooks/
│   └── 01_train_models_colab.ipynb       ← trains helmet + plate detectors on Colab T4
│
├── tests/                                ← test_preprocessing.py, test_violations.py, test_ocr.py
└── docs/
    ├── Traffic_Violation_Final_Implementation_Plan.md  ← PRIMARY source of truth
    ├── COLAB_GUIDE.md                                  ← training workflow
    └── context.md                                      ← this file
```

---

## Key Data Structures (`src/models.py`)

```python
@dataclass
class Detection:
    class_name: str; confidence: float; bbox: tuple; frame_id: int

@dataclass
class TrackedObject:
    track_id: int; class_name: str; bbox: tuple; confidence: float
    frame_id: int; centroid_history: list[tuple[int, int]]   # last 60 (cx, cy)

@dataclass
class ViolationRecord:
    violation_type: str          # "helmet", "triple_riding", "illegal_parking", ...
    confidence: float; vehicle_id: int; bbox: tuple
    timestamp: str; frame_id: int
    plate_number: str | None; plate_confidence: float | None
    status: str                  # "auto_flagged" | "review" | "indeterminate"
    evidence_image_path: str | None; evidence_json_path: str | None
    is_blurry: bool; camera_id: str
```

---

## Configuration Notes

- `configs/pipeline.yaml` — `models.vehicle_detector` = `yolo11s.pt`; `inference.device` = `cuda`.
- `configs/violations.yaml` — triple riding key is `min_person_overlap_ratio` (0.5); parking section is keyed `illegal_parking`; helmet has `flag_invalid_helmet` (default false).
- `configs/cameras.yaml` — set zones with `python scripts/draw_zones.py --video <clip>`.

---

## How to Run

```powershell
cd Traffic-Violations-Using-Computer-Vision
.\venv\Scripts\activate

python app.py --video data\samples\test_video.mp4 --show   # full pipeline (GPU)
python app.py --video 0                                     # webcam
python app.py --video data\samples\test_video.mp4 --dry-run # preprocessing only (no ML stack needed)
python app.py --dashboard                                   # Streamlit on :8501
python -m pytest tests\ -q                                  # tests
```

### Docker
```bash
docker compose build
docker compose up dashboard                                           # UI on :8501
docker compose run --rm pipeline python app.py --video data/samples/test_video.mp4
```

---

## Setup From Scratch

```powershell
# 1. Python 3.11 venv (NOT 3.13 — paddle/torch have no 3.13 wheels)
py -3.11 -m venv venv; .\venv\Scripts\activate

# 2. GPU torch first, then the rest
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 3. Vehicle weights (helmet/plate are trained on Colab — see docs/COLAB_GUIDE.md)
python scripts/download_models.py
# place helmet_yolov8.pt + plate_yolov8.pt in models/weights/
```

Training: `scripts/download_datasets.py` pulls datasets → `notebooks/01_train_models_colab.ipynb` trains on Colab T4 → download `best.pt` weights locally. Vehicle/person detection uses pretrained COCO YOLO (NOT fine-tuned — fine-tuning on a vehicle-only set would erase the `person` class that triple-riding and helmet association need).

---

## Bugs Fixed (history, so they don't regress)

- `metrics.py`: `np.trapz` → `np.trapezoid` (removed in NumPy 2.x); `average_precision` returns 0 (not phantom 0.5) when there are no predictions.
- `triple_riding.py`: IoU → containment (intersection / person-area).
- `tracker.py`: replaced private ultralytics `BYTETracker` call (broke on 8.4.x) with a self-contained IoU tracker.
- `helmet.py`: full-frame detection + motorcycle association instead of a fixed top-fraction head crop (which missed the head).
- `plate_reader.py`: PaddleOCR → EasyOCR (paddle 3.x oneDNN runtime bug; EasyOCR uses the same CUDA GPU).
- `video_pipeline.py`: lazy ML imports so `--dry-run` works without torch/easyocr.
- config: parking key `parking` → `illegal_parking`; triple key → `min_person_overlap_ratio`.

---

## Known Limitations (Documented Honestly)

- **Seatbelt**: `indeterminate` until the binary CNN is trained (correct behaviour, not a bug).
- **Auto-rickshaw**: COCO has no such class; falls under `motorcycle`/`car`. Fine-tune on IDD if needed.
- **Rain handling**: classical filter only (median blur + unsharp), not a deraining network.
- **Signal detection**: HSV-based; may fail under extreme overexposure / nighttime.
- **Parking dwell timer**: resets on track ID switch under occlusion (tracker limitation).

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Vehicle/person detection | YOLO11s (Ultralytics) — pretrained COCO |
| Plate detection | YOLO — fine-tuned, class `License_Plate` |
| Helmet detection | YOLO — fine-tuned, full-frame + association |
| Seatbelt classification | Custom `_SeatbeltCNN` (PyTorch) |
| Tracking | Self-contained IoU tracker |
| OCR | EasyOCR (GPU via torch) |
| Preprocessing | OpenCV (CLAHE, Laplacian, median blur) |
| Database | SQLite via SQLAlchemy (Postgres-ready) |
| Dashboard | Streamlit |
| Deployment | Docker + docker-compose |
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

## Commit Hygiene

`.gitignore` excludes: `.pt`/`.pth`/`.onnx` weights, `.mp4` videos, SQLite `.db`/`.db-journal`/`.db-wal`, `venv/`, `artifacts/evidence/`, logs, caches. Source, configs, `app.py`, `requirements.txt`, docs, the notebook, Dockerfile/compose are committed. 59 tracked files; 0 files >1MB.

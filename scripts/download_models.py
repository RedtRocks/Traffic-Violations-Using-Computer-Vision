"""
Download required pretrained model weights.
Run this once after setting up the environment.

Downloads:
  - YOLOv8m (vehicle detection) — pretrained COCO weights
  - YOLOv8n for helmet (placeholder — replace with fine-tuned weights)
  - Indian plate detection model (placeholder — replace with fine-tuned weights)

Note: Helmet and plate models require domain-specific fine-tuning.
      Replace the placeholder URLs with your trained model URLs or paths.
"""

import sys
import urllib.request
from pathlib import Path


MODELS_DIR = Path(__file__).parent.parent / "models" / "weights"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


DOWNLOADS = [
    {
        "name": "YOLOv8m (vehicle detection)",
        "filename": "yolov8m.pt",
        "source": "ultralytics",     # downloaded via ultralytics auto-download
    },
    {
        "name": "YOLOv8n (helmet classifier placeholder)",
        "filename": "helmet_yolov8.pt",
        "source": "ultralytics",
        "note": "Replace with your fine-tuned helmet model weights.",
    },
]


def download_via_ultralytics(filename: str):
    from ultralytics import YOLO
    target = MODELS_DIR / filename
    if target.exists():
        print(f"  Already exists: {target}")
        return
    # Ultralytics auto-downloads to its cache; copy to our weights dir
    model = YOLO(filename.replace("_yolov8.pt", "8").replace("yolov8", "yolov8"))
    import shutil
    from ultralytics.utils import WEIGHTS_DIR
    cached = WEIGHTS_DIR / filename
    if cached.exists():
        shutil.copy(cached, target)
        print(f"  Saved to: {target}")


def main():
    print("Downloading model weights...\n")
    for entry in DOWNLOADS:
        print(f"[{entry['name']}]")
        if entry.get("note"):
            print(f"  NOTE: {entry['note']}")
        if entry["source"] == "ultralytics":
            download_via_ultralytics(entry["filename"])
        print()

    print("Done. Update configs/pipeline.yaml model paths if needed.")
    print("For fine-tuned weights (helmet, plate), place .pt files in models/weights/ manually.")


if __name__ == "__main__":
    main()

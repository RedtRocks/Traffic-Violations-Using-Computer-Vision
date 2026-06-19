"""
License plate OCR using PaddleOCR.
Accepts a plate crop (numpy BGR array) and returns the recognized text and confidence.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class PlateReadResult:
    text: str
    confidence: float
    is_partial: bool


class PlateReader:
    def __init__(self, lang: str = "en", use_gpu: bool = False):
        from paddleocr import PaddleOCR
        self._ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu, show_log=False)

    def read(self, plate_crop: np.ndarray) -> PlateReadResult | None:
        if plate_crop is None or plate_crop.size == 0:
            return None

        results = self._ocr.ocr(plate_crop, cls=True)
        if not results or not results[0]:
            return None

        texts = []
        confidences = []
        for line in results[0]:
            if line and len(line) >= 2:
                text_conf = line[1]
                if text_conf:
                    texts.append(text_conf[0])
                    confidences.append(float(text_conf[1]))

        if not texts:
            return None

        combined_text = " ".join(texts).upper().strip()
        avg_conf = sum(confidences) / len(confidences)
        is_partial = len(combined_text.replace(" ", "")) < 4

        return PlateReadResult(
            text=combined_text,
            confidence=avg_conf,
            is_partial=is_partial,
        )

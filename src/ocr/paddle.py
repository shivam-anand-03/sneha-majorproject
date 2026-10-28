"""PP-OCRv5 recognition (PaddleOCR 3.x) on pre-cropped lines. Baseline engine."""
from __future__ import annotations

import time

import numpy as np

from src.common.config import Config
from src.ocr.base import OCRLine


class PaddleRec:
    name = "paddle"

    def __init__(self, cfg: Config):
        from paddleocr import TextRecognition

        self.cfg = cfg
        self.model = TextRecognition(model_name=cfg.ocr.paddle.rec_model)

    def recognize(self, crops: list[np.ndarray]) -> list[OCRLine]:
        out = []
        for c in crops:
            t0 = time.time()
            img = np.stack([c] * 3, -1) if c.ndim == 2 else c
            r = self.model.predict(img)[0]
            text, score = r["rec_text"], float(r["rec_score"])
            thr = self.cfg.ocr.confidence.word_low_threshold
            # PP-OCR gives one score per line; propagate to words so low lines flag every word
            words = [{"word": w, "conf": round(score, 4), "low": score < thr} for w in text.split()]
            out.append(OCRLine(text=text, engine=self.name, confidence=round(score, 4), words=words,
                               candidates={self.name: text}, seconds=time.time() - t0))
        return out

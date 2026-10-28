"""OCR engine factory + confidence routing.

Primary engine reads every line. Lines below the line-confidence threshold are re-read by
the fallback engine (local VLM or Gemini). The primary text stays as the default output and both
readings are kept in OCRLine.candidates; the line is always flagged for teacher review.
The fallback reading is never auto-trusted.
"""
from __future__ import annotations

import numpy as np

from src.common.config import Config
from src.ocr.base import OCRLine

_ENGINES: dict[str, object] = {}


def get_engine(name: str, cfg: Config):
    if name not in _ENGINES:
        if name == "trocr":
            from src.ocr.trocr import TrOCR

            _ENGINES[name] = TrOCR(cfg)
        elif name == "paddle":
            from src.ocr.paddle import PaddleRec

            _ENGINES[name] = PaddleRec(cfg)
        elif name in ("vlm", "gemini"):
            from src.ocr.vlm import VLMRec

            _ENGINES[name] = VLMRec(cfg, name)
        else:
            raise ValueError(f"unknown OCR engine {name}")
    return _ENGINES[name]


def _flag(line: OCRLine, cfg: Config) -> OCRLine:
    c = cfg.ocr.confidence
    if line.confidence is not None and line.confidence < c.line_low_threshold:
        line.needs_review = True
        line.review_reasons.append(f"line confidence {line.confidence:.2f} < {c.line_low_threshold}")
    low = [w["word"] for w in line.words if w.get("low")]
    if low:
        line.needs_review = True
        line.review_reasons.append(f"low-confidence words: {', '.join(low[:6])}")
    return line


def recognize_lines(crops: list[np.ndarray], cfg: Config, engine: str | None = None,
                    use_fallback: bool | None = None) -> list[OCRLine]:
    engine = engine or cfg.ocr.engine
    lines = [_flag(l, cfg) for l in get_engine(engine, cfg).recognize(crops)]
    use_fallback = cfg.ocr.confidence.use_vlm_fallback if use_fallback is None else use_fallback
    fb_name = cfg.ocr.confidence.fallback_engine
    if not use_fallback or fb_name == engine:
        return lines
    low_idx = [i for i, l in enumerate(lines)
               if l.confidence is not None and l.confidence < cfg.ocr.confidence.line_low_threshold]
    if not low_idx:
        return lines
    fb = get_engine(fb_name, cfg)
    if hasattr(fb, "available") and not fb.available():
        for i in low_idx:
            lines[i].review_reasons.append(f"fallback {fb_name} unavailable")
        return lines
    fb_lines = fb.recognize([crops[i] for i in low_idx], hard=True)
    for i, fl in zip(low_idx, fb_lines):
        lines[i].candidates[fb_name] = fl.text
        lines[i].needs_review = True
        lines[i].review_reasons.append(f"low confidence: {fb_name} re-read kept alongside; teacher to confirm")
        lines[i].seconds += fl.seconds
    return lines

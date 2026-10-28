"""VLM transcribers: local Qwen2.5-VL or Gemini, both via the shared LLM client.

VLMs give no calibrated token confidence here, so confidence=None and every line they
produce as a *fallback* is marked for review (never auto-trusted).
"""
from __future__ import annotations

import time

import numpy as np

from src.common.config import Config
from src.common.llm_client import LLMUnavailable, get_client
from src.common.prompts import OCR_LINE_SCHEMA, ocr_line_prompt
from src.ocr.base import OCRLine


def _pad_for_vlm(crop: np.ndarray) -> np.ndarray:
    """Very short line crops get padded so the VLM sees enough context pixels."""
    h, w = crop.shape[:2]
    if h >= 32:
        return crop
    pad = (32 - h) // 2 + 1
    fill = 255 if crop.ndim == 2 else (255, 255, 255)
    import cv2

    return cv2.copyMakeBorder(crop, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=fill)


class VLMRec:
    """engine='vlm' -> local Qwen (smaller fallback model on failure); engine='gemini' -> Gemini models."""

    def __init__(self, cfg: Config, engine: str = "vlm"):
        self.cfg = cfg
        self.name = engine
        self.client = get_client(cfg)
        if engine == "gemini":
            self.strategies = [f"gemini:{m}" for m in cfg.ocr.gemini.models]
            self.max_tokens = cfg.ocr.gemini.max_output_tokens
        else:
            self.strategies = [f"local-vl:{cfg.ocr.vlm.model}", f"local-vl:{cfg.ocr.vlm.fallback_model}"]
            self.max_tokens = cfg.ocr.vlm.max_new_tokens

    def available(self) -> bool:
        return self.client.gemini_enabled() if self.name == "gemini" else True

    def recognize(self, crops: list[np.ndarray], hard: bool = False) -> list[OCRLine]:
        strategies = self.strategies
        if hard and self.name == "gemini":
            # hard lines go straight to the stronger model
            strategies = self.strategies[self.cfg.ocr.gemini.hard_line_model_index:]
        out = []
        for c in crops:
            img = np.stack([c] * 3, -1) if c.ndim == 2 else c
            t0 = time.time()
            try:
                res = self.client.structured_call(ocr_line_prompt(), OCR_LINE_SCHEMA, strategies, images=[_pad_for_vlm(img)],
                                                  max_tokens=self.max_tokens, purpose=f"ocr:{self.name}")
                text, ill, model = str(res.data["text"]).strip(), bool(res.data["illegible"]), res.model
            except LLMUnavailable as e:
                text, ill, model = "", True, f"unavailable: {e}"[:120]
            reasons = ["illegible per VLM"] if ill else []
            out.append(OCRLine(text=text, engine=f"{self.name}({model})", confidence=None,
                               words=[{"word": w, "conf": None, "low": "[?]" in w} for w in text.split()],
                               candidates={self.name: text}, needs_review=ill, review_reasons=reasons,
                               seconds=time.time() - t0))
        return out

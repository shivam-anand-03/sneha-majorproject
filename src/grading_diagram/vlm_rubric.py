"""Semantic diagram check by a VLM against a rubric (components, labels, arrows, structure).

The raw VLM JSON is always kept. If no VLM is available the result is None with a reason;
scores are never fabricated.
"""
from __future__ import annotations

import numpy as np

from src.common.config import Config
from src.common.llm_client import LLMUnavailable, get_client
from src.common.prompts import DIAGRAM_RUBRIC_SCHEMA, diagram_rubric_prompt


def strategies(cfg: Config) -> list[str]:
    if cfg.grading_diagram.vlm_rubric.backend == "gemini":
        return [f"gemini:{m}" for m in cfg.ocr.gemini.models]
    return [f"local-vl:{cfg.ocr.vlm.model}", f"local-vl:{cfg.ocr.vlm.fallback_model}"]


def _rgb(img: np.ndarray) -> np.ndarray:
    return np.stack([img] * 3, -1) if img.ndim == 2 else img


def rubric_check(student: np.ndarray, reference: np.ndarray, rubric: str, cfg: Config) -> tuple[dict | None, str]:
    """Returns (result, reason). result has per-criterion score/reason, mean score, missing, raw, model."""
    v = cfg.grading_diagram.vlm_rubric
    criteria = list(v.criteria)
    try:
        res = get_client(cfg).structured_call(
            diagram_rubric_prompt(rubric, criteria), DIAGRAM_RUBRIC_SCHEMA, strategies(cfg),
            images=[_rgb(reference), _rgb(student)], max_tokens=v.max_new_tokens, purpose="diagram_rubric")
    except LLMUnavailable as e:
        return None, f"VLM unavailable: {str(e)[:160]}"
    raw = res.data
    sub = {}
    for c in criteria:
        item = raw.get(c, {}) or {}
        try:
            s = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            s = 0.0
        sub[c] = {"score": float(np.clip(s, 0.0, 1.0)), "reason": str(item.get("reason", ""))}
    return {
        "criteria": sub,
        "score": float(np.mean([sub[c]["score"] for c in criteria])),
        "missing_elements": [str(m) for m in raw.get("missing_elements", [])],
        "raw": raw,
        "model": res.model,
    }, "ok"

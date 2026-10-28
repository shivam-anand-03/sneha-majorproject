"""The only place history may touch marks, and only when cfg.history.use_history_in_score is true."""
from __future__ import annotations

from src.common.config import Config


def apply_history(model_score_norm: float, expected_norm: float | None, cfg: Config) -> dict:
    """Scores are normalised 0..1. Default config: the model's score is returned untouched."""
    used = bool(cfg.history.use_history_in_score) and expected_norm is not None
    b = float(cfg.history.history_blend)
    marks = (1 - b) * model_score_norm + b * expected_norm if used else model_score_norm
    return {"marks": float(marks), "history_used": used, "model_marks": float(model_score_norm),
            "expected": None if expected_norm is None else float(expected_norm)}

"""History lookup for one graded answer: expected score, anomaly flag, weak topics, feedback."""
from __future__ import annotations

from src.common.config import Config
from src.history.feedback import personalized_feedback
from src.history.model import HistoryModel, anomaly_check, build_training_frame, expected_score, fit_history_model
from src.history.store import history_path, load_history, student_profile

_CACHE: dict[tuple, tuple] = {}


def _load(cfg: Config):
    """History table + fitted model, refit only when the CSV changes."""
    p = history_path(cfg)
    key = (str(p), p.stat().st_mtime if p.exists() else None)
    if key not in _CACHE:
        hist = load_history(cfg)
        hm = fit_history_model(build_training_frame(hist, cfg), cfg) if len(hist) else None
        _CACHE.clear()
        _CACHE[key] = (hist, hm)
    return _CACHE[key]


def history_for_result(student_id_anon: str, topic: str | None, current_score_norm: float,
                       concept_table: list[dict], cfg: Config) -> dict:
    hist, hm = _load(cfg)
    prof = student_profile(hist, student_id_anon, cfg) if len(hist) else {"n_attempts": 0, "weak_topics": []}
    expected = None
    if hm is not None and prof["n_attempts"]:
        expected = (expected_score(hm, hist, student_id_anon, topic, cfg) if topic
                    else prof["overall_mean"])
    an = anomaly_check(current_score_norm, expected, hm or HistoryModel("mean", None, 0.2, 0.5), cfg)
    return {"expected": expected, "anomaly": an["flag"], "z": an["z"], "anomaly_reason": an["reason"],
            "weak_topics": prof.get("weak_topics", []),
            "feedback": personalized_feedback(prof, concept_table, topic, cfg)}

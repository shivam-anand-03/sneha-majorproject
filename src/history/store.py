"""Student learning-history table and per-student features.

Schema: student_id_anonymized, topic, question_id, date (ISO), score_norm (0..1), attempt_no
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import Config, resolve

COLUMNS = ["student_id_anonymized", "topic", "question_id", "date", "score_norm", "attempt_no"]


def history_path(cfg: Config) -> Path:
    return resolve(cfg, cfg.paths.history)


def load_history(cfg: Config, path: str | Path | None = None) -> pd.DataFrame:
    p = Path(path) if path else history_path(cfg)
    if not p.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(p)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"history csv missing columns {missing}")
    return df[COLUMNS].sort_values(["student_id_anonymized", "date", "attempt_no"]).reset_index(drop=True)


def append_history(rows: list[dict], cfg: Config, path: str | Path | None = None) -> pd.DataFrame:
    p = Path(path) if path else history_path(cfg)
    df = pd.concat([load_history(cfg, p), pd.DataFrame(rows, columns=COLUMNS)], ignore_index=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return df


def trend_slope(scores: list[float] | np.ndarray, window: int) -> float:
    """Least-squares slope of the last `window` scores (score change per attempt)."""
    s = np.asarray(scores, float)[-window:]
    if len(s) < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(s)), s, 1)[0])


def student_profile(hist: pd.DataFrame, student: str, cfg: Config) -> dict:
    h = hist[hist.student_id_anonymized == student].sort_values(["date", "attempt_no"])
    if h.empty:
        return {"student": student, "n_attempts": 0, "overall_mean": None, "topic_means": {},
                "trend": 0.0, "weak_topics": []}
    tm = h.groupby("topic").score_norm.mean().round(3).to_dict()
    thr = cfg.history.weak_topic_threshold
    return {
        "student": student,
        "n_attempts": int(len(h)),
        "overall_mean": round(float(h.score_norm.mean()), 3),
        "topic_means": tm,
        "trend": round(trend_slope(h.score_norm.to_numpy(), cfg.history.trend_window), 4),
        "weak_topics": sorted([t for t, m in tm.items() if m < thr], key=lambda t: tm[t]),
    }


def pair_features(hist: pd.DataFrame, student: str, topic: str, cfg: Config,
                  topic_means: dict[str, float] | None = None) -> dict[str, float]:
    """Features for predicting the next score of `student` on `topic` from prior history only."""
    h = hist[hist.student_id_anonymized == student].sort_values(["date", "attempt_no"])
    ht = h[h.topic == topic]
    glob = float(hist.score_norm.mean()) if len(hist) else 0.5
    tmeans = topic_means if topic_means is not None else hist.groupby("topic").score_norm.mean().to_dict()
    return {
        "student_mean": float(h.score_norm.mean()) if len(h) else glob,
        "student_topic_mean": float(ht.score_norm.mean()) if len(ht) else np.nan,
        "student_last": float(h.score_norm.iloc[-1]) if len(h) else glob,
        "n_attempts": float(len(h)),
        "n_topic_attempts": float(len(ht)),
        "trend": trend_slope(h.score_norm.to_numpy(), cfg.history.trend_window),
        "topic_difficulty": float(tmeans.get(topic, glob)),
    }

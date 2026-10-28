"""Expected-score model from history (LightGBM, ridge/mean fallback) + anomaly flags.

The anomaly flag only routes an answer to teacher review. It never changes marks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.common.config import Config
from src.history.store import pair_features

FEATURES = ["student_mean", "student_topic_mean", "student_last", "n_attempts", "n_topic_attempts", "trend",
            "topic_difficulty"]
MIN_ROWS_LGBM = 50


@dataclass
class HistoryModel:
    kind: str
    model: object | None
    resid_std: float
    global_mean: float
    feature_names: list[str] = field(default_factory=lambda: list(FEATURES))

    def predict_frame(self, X: pd.DataFrame) -> np.ndarray:
        if self.kind == "mean" or self.model is None:
            return X["student_mean"].fillna(self.global_mean).to_numpy()
        Xf = X[self.feature_names]
        if self.kind == "ridge":
            Xf = _impute(Xf, self.global_mean)
        return np.clip(self.model.predict(Xf), 0.0, 1.0)


def _impute(X: pd.DataFrame, global_mean: float) -> pd.DataFrame:
    X = X.copy()
    X["student_topic_mean"] = X["student_topic_mean"].fillna(X["student_mean"])
    return X.fillna(global_mean)


def build_training_frame(hist: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """For every attempt, features from that student's strictly earlier attempts -> target score."""
    rows = []
    hist = hist.sort_values(["student_id_anonymized", "date", "attempt_no"])
    tmeans = hist.groupby("topic").score_norm.mean().to_dict()
    for sid, h in hist.groupby("student_id_anonymized"):
        h = h.reset_index(drop=True)
        for i in range(1, len(h)):
            f = pair_features(h.iloc[:i], sid, h.topic[i], cfg, tmeans)
            rows.append({**f, "target": h.score_norm[i], "student": sid, "topic": h.topic[i]})
    return pd.DataFrame(rows)


def fit_history_model(train: pd.DataFrame, cfg: Config, kind: str = "auto") -> HistoryModel:
    gm = float(train.target.mean()) if len(train) else 0.5
    if kind == "auto":
        kind = "lgbm" if len(train) >= MIN_ROWS_LGBM else ("ridge" if len(train) >= 10 else "mean")
    model = None
    if kind == "lgbm":
        import lightgbm as lgb

        p = cfg.history.lgbm
        model = lgb.LGBMRegressor(n_estimators=p.n_estimators, learning_rate=p.learning_rate,
                                  num_leaves=p.num_leaves, min_child_samples=p.min_child_samples,
                                  random_state=cfg.project.seed, verbose=-1)
        model.fit(train[FEATURES], train.target)
    elif kind == "ridge":
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=1.0).fit(_impute(train[FEATURES], gm), train.target)
    hm = HistoryModel(kind, model, 0.0, gm)
    resid = train.target.to_numpy() - hm.predict_frame(train)
    hm.resid_std = float(max(resid.std(), 1e-3)) if len(resid) > 1 else 0.2
    return hm


def expected_score(hm: HistoryModel, hist: pd.DataFrame, student: str, topic: str, cfg: Config) -> float | None:
    if not (hist.student_id_anonymized == student).any():
        return None
    X = pd.DataFrame([pair_features(hist, student, topic, cfg)])
    return float(hm.predict_frame(X)[0])


def anomaly_check(actual: float, expected: float | None, hm: HistoryModel, cfg: Config) -> dict:
    """Flag for teacher review when the awarded score is far from history. Never a penalty."""
    if expected is None:
        return {"flag": False, "z": None, "reason": "no history for this student"}
    z = (actual - expected) / hm.resid_std
    flag = abs(z) > cfg.history.anomaly_z
    reason = (f"score {actual:.2f} differs from history-expected {expected:.2f} (z={z:+.1f}); "
              "teacher review suggested, marks unchanged") if flag else ""
    return {"flag": bool(flag), "z": round(float(z), 2), "reason": reason}

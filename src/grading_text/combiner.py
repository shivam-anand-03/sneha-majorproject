"""Learned combination of text-grading signals (ridge regression fitted on human marks).

Anti keyword-stuffing: concept coverage is gated by the cross-encoder. A list of correct
terms with wrong meaning gets high coverage but low cross-encoder score, so its coverage
contribution is scaled down by ce / gate.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.common.config import Config, resolve


def gated_coverage(coverage: np.ndarray, ce: np.ndarray, gate: float) -> np.ndarray:
    coverage, ce = np.asarray(coverage, float), np.asarray(ce, float)
    return coverage * np.clip(ce / gate, 0.0, 1.0)


def feature_matrix(feats: dict[str, np.ndarray], cfg: Config) -> np.ndarray:
    c = cfg.grading_text.combiner
    cols = []
    for name in c.features:
        v = np.asarray(feats[name], float)
        if name == "concept_coverage":
            v = gated_coverage(v, feats["cross_encoder"], c.coverage_gate_ce_min)
        cols.append(v)
    return np.stack(cols, 1)


def fit(feats: dict[str, np.ndarray], y: np.ndarray, cfg: Config) -> dict:
    from sklearn.linear_model import Ridge

    X = feature_matrix(feats, cfg)
    m = Ridge(alpha=cfg.grading_text.combiner.alpha, positive=True).fit(X, y)
    w = {"features": list(cfg.grading_text.combiner.features), "coef": m.coef_.tolist(),
         "intercept": float(m.intercept_), "n_train": int(len(y))}
    path = resolve(cfg, cfg.grading_text.combiner.weights_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(w, indent=2))
    return w


def load_weights(cfg: Config, path: str | Path | None = None) -> dict:
    p = Path(path) if path else resolve(cfg, cfg.grading_text.combiner.weights_path)
    if not p.exists():
        raise FileNotFoundError(f"combiner weights missing: {p} (run scripts/train_text_grader.py)")
    return json.loads(p.read_text())


def predict(feats: dict[str, np.ndarray], cfg: Config, weights: dict | None = None) -> np.ndarray:
    w = weights or load_weights(cfg)
    X = feature_matrix(feats, cfg)
    return np.clip(X @ np.asarray(w["coef"]) + w["intercept"], 0.0, 1.0)

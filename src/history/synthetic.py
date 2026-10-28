"""Seeded synthetic learning histories (latent ability + topic difficulty + noise).

Used only because no real history data exists. Abilities can be anchored to observed
scores so the ablation has a realistic link between past and present performance.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from src.history.store import COLUMNS

DEFAULT_TOPICS = [f"E{i:02d}" for i in range(1, 13)]


def make_history(students: list[str], seed: int, topics: list[str] | None = None,
                 ability: dict[str, float] | None = None, n_attempts: tuple[int, int] = (8, 20),
                 noise: float = 0.15) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    topics = topics or DEFAULT_TOPICS
    difficulty = dict(zip(topics, rng.normal(0, 0.12, len(topics))))
    start = dt.date(2025, 7, 1)
    rows = []
    for s in students:
        a = ability[s] if ability and s in ability else float(np.clip(rng.normal(0.65, 0.15), 0.05, 0.98))
        drift = rng.normal(0, 0.01)
        n = int(rng.integers(n_attempts[0], n_attempts[1] + 1))
        for k in range(n):
            t = topics[int(rng.integers(len(topics)))]
            sc = np.clip(a - difficulty[t] + drift * k + rng.normal(0, noise), 0, 1)
            rows.append({"student_id_anonymized": s, "topic": t, "question_id": f"{t}.H{k:02d}",
                         "date": (start + dt.timedelta(days=7 * k + int(rng.integers(0, 5)))).isoformat(),
                         "score_norm": round(float(sc), 3), "attempt_no": k + 1})
    return pd.DataFrame(rows, columns=COLUMNS)

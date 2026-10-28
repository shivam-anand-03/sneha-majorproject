"""Full text grader: embedding similarity + cross-encoder + concept coverage -> combined score."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.common.config import Config
from src.grading_text import combiner, concepts, cross_encoder, embed


@dataclass
class TextGrade:
    score: float                        # 0..1 combined
    features: dict[str, float]
    concept_table: list[dict]
    weights: dict
    contributions: dict[str, float] = field(default_factory=dict)


def grade_text(question: str, reference: str, answer: str, cfg: Config,
               key_concepts: list[str] | None = None) -> TextGrade:
    answer = answer or ""
    if not answer.strip():
        return TextGrade(0.0, {"embed_sim": 0.0, "cross_encoder": 0.0, "concept_coverage": 0.0}, [], {}, {})
    es = float(embed.pair_similarity([reference], [answer], cfg)[0])
    ce = float(cross_encoder.predict([question], [reference], [answer], cfg)[0])
    cons = key_concepts or concepts.extract_concepts(reference, cfg)
    table = concepts.match_concepts(cons, answer, cfg)
    cov = concepts.coverage(table)
    feats = {"embed_sim": es, "cross_encoder": ce, "concept_coverage": cov}
    w = combiner.load_weights(cfg)
    X = combiner.feature_matrix({k: np.array([v]) for k, v in feats.items()}, cfg)[0]
    contrib = {f: float(c * x) for f, c, x in zip(w["features"], w["coef"], X)}
    contrib["intercept"] = w["intercept"]
    score = float(np.clip(sum(contrib.values()), 0, 1))
    return TextGrade(score, feats, table, w, contrib)

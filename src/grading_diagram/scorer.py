"""Diagram score = w_sim * transformed embedding similarity + w_vlm * VLM rubric score."""
from __future__ import annotations

import numpy as np

from src.common.config import Config
from src.grading_diagram import embed_sim, vlm_rubric


def grade_diagram(student: np.ndarray | None, reference: np.ndarray, rubric: str, cfg: Config,
                  use_vlm: bool | None = None, kinds: list[str] | None = None) -> dict:
    g = cfg.grading_diagram
    if student is None or student.size == 0:
        return {"sim": {}, "sim_blend": 0.0, "sim_transformed": 0.0, "vlm": None, "score": 0.0,
                "disagreement": None, "needs_review": True, "reasons": ["no diagram found on the sheet"]}
    use_vlm = g.vlm_rubric.enabled if use_vlm is None else use_vlm
    sims = embed_sim.similarities(student, reference, cfg, kinds)
    sb = embed_sim.blend(sims, g.embed_weights)
    st = float(embed_sim.transform01(sb, g.transform.k, g.transform.c))
    reasons: list[str] = []
    vlm, why = (vlm_rubric.rubric_check(student, reference, rubric, cfg) if use_vlm else (None, "VLM disabled"))
    if vlm is None:
        score, dis = st, None
        reasons.append(f"embedding similarity only ({why})")
        needs_review = bool(use_vlm)  # VLM was wanted but missing -> review
    else:
        w = g.blend
        score = float((w.w_sim * st + w.w_vlm * vlm["score"]) / (w.w_sim + w.w_vlm))
        dis = float(abs(st - vlm["score"]))
        needs_review = dis > cfg.xai.review.diagram_disagreement
        if needs_review:
            reasons.append(f"embedding ({st:.2f}) and VLM ({vlm['score']:.2f}) disagree by {dis:.2f}")
    return {"sim": sims, "sim_blend": sb, "sim_transformed": st, "vlm": vlm,
            "score": float(np.clip(score, 0, 1)), "disagreement": dis,
            "needs_review": needs_review, "reasons": reasons}

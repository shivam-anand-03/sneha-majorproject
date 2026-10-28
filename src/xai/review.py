"""'Teacher review needed' decision from a pipeline result dict."""
from __future__ import annotations

from src.common.config import Config


def _ocr_reasons(lines: list[dict], thr: float) -> list[str]:
    confs = [l["confidence"] for l in lines if l.get("confidence") is not None]
    out = []
    if confs and min(confs) < thr:
        out.append(f"low OCR confidence (min line {min(confs):.2f} < {thr})")
    flagged = sum(bool(l.get("needs_review")) for l in lines)
    if flagged:
        out.append(f"{flagged} OCR line(s) flagged for checking")
    return out


def _diagram_reasons(diagrams: list[dict], cfg: Config) -> list[str]:
    out = []
    thr = cfg.xai.review.diagram_disagreement
    for i, d in enumerate(diagrams):
        vlm = d.get("vlm")
        if vlm is None:
            if cfg.grading_diagram.vlm_rubric.enabled:
                out.append(f"diagram {i}: rubric check (VLM) unavailable")
        elif d.get("sim_transformed") is not None and abs(d["sim_transformed"] - vlm["score"]) > thr:
            out.append(f"diagram {i}: scores disagree (visual {d['sim_transformed']:.2f} vs rubric {vlm['score']:.2f})")
    return out


def needs_review(result: dict, cfg: Config) -> tuple[bool, list[str]]:
    r = cfg.xai.review
    reasons = _ocr_reasons(result.get("lines") or [], r.ocr_conf_threshold)
    feats = (result.get("text_grade") or {}).get("features") or {}
    if "embed_sim" in feats and "cross_encoder" in feats:
        if abs(feats["embed_sim"] - feats["cross_encoder"]) > r.model_disagreement:
            reasons.append(f"text models disagree (similarity {feats['embed_sim']:.2f} vs "
                           f"cross-encoder {feats['cross_encoder']:.2f})")
    reasons += _diagram_reasons(result.get("diagrams") or [], cfg)
    if (result.get("history") or {}).get("anomaly"):
        reasons.append("score is unusual for this student's history (anomaly flag)")
    return bool(reasons), reasons

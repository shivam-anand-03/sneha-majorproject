"""Demo pipeline for the app when the full model pipeline is not installed/trained.

Page preprocessing, line segmentation and diagram detection are REAL (src.pipeline.page).
OCR text, grading signals and diagram scores are placeholder values derived from the
answer key, clearly labelled with engine "demo" so nobody mistakes them for model output.
"""
from __future__ import annotations

import random
import re
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from src.common.config import Config
from src.pipeline.page import analyze_page
from src.preprocess.pdf import load_pages

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]+")
PAGE_MAX_W = 1400


class BadImage(ValueError):
    pass


def _save_png(path: Path, img: np.ndarray | None) -> None:
    if img is None or not cv2.imwrite(str(path), img):
        raise BadImage(f"could not write {path.name}")


def _demo_lines(n: int, reference: str) -> list[str]:
    """Spread the reference answer over the detected lines (placeholder transcription)."""
    words = reference.split() or ["(no", "reference)"]
    per = max(1, len(words) // max(n, 1))
    out = []
    for i in range(n):
        chunk = words[i * per:(i + 1) * per] if i < n - 1 else words[(n - 1) * per:]
        out.append(" ".join(chunk) if chunk else "")
    return out


def _line_record(i: int, box, crop_name: str, text: str, rng: random.Random, cfg: Config, edited: bool) -> dict:
    thr = cfg.ocr.confidence.word_low_threshold
    words = [{"word": w, "conf": round(rng.uniform(0.45, 0.99), 3)} for w in text.split()]
    for w in words:
        w["low"] = w["conf"] < thr
    conf = round(float(np.mean([w["conf"] for w in words])) if words else 0.0, 3)
    low = [w["word"] for w in words if w["low"]]
    reasons = [] if edited else ([f"low-confidence words: {', '.join(low[:5])}"] if low else [])
    cands = {"demo": text}
    if low and not edited:
        cands["demo-alt"] = " ".join(w["word"].lower() for w in words)
    return {"idx": i, "bbox": [int(v) for v in box], "crop": crop_name, "text": text, "candidates": cands,
            "confidence": None if edited else conf, "words": words, "needs_review": bool(reasons),
            "review_reasons": reasons, "edited": edited}


def _concepts(reference: str, answer: str) -> list[dict]:
    seen, cons = set(), []
    for w in _WORD.findall(reference.lower()):
        if len(w) > 4 and w not in seen:
            seen.add(w)
            cons.append(w)
    ans = set(_WORD.findall(answer.lower()))
    rows = []
    for c in cons[:7]:
        if c in ans:
            rows.append({"concept": c, "status": "matched", "match": c, "method": "lemma", "score": 1.0})
        elif any(a[:4] == c[:4] for a in ans):
            m = next(a for a in ans if a[:4] == c[:4])
            rows.append({"concept": c, "status": "partial", "match": m, "method": "fuzzy(demo)", "score": 0.5})
        else:
            rows.append({"concept": c, "status": "missing", "match": "", "method": "", "score": 0.0})
    return rows


def _text_grade(reference: str, answer: str) -> dict:
    table = _concepts(reference, answer)
    cov = float(np.mean([r["score"] for r in table])) if table else 0.0
    feats = {"embed_sim": round(0.35 + 0.6 * cov, 3), "cross_encoder": round(0.2 + 0.7 * cov, 3),
             "concept_coverage": round(cov, 3)}
    contrib = {"embed_sim": round(0.2 * feats["embed_sim"], 3), "cross_encoder": round(0.55 * feats["cross_encoder"], 3),
               "concept_coverage": round(0.25 * cov, 3), "intercept": 0.0}
    key = {r["concept"] for r in table if r["status"] != "missing"}
    attrs = [{"word": w, "attr": round(0.6 if w.lower().strip(".,;") in key else -0.05, 3)} for w in answer.split()]
    return {"score": round(min(1.0, sum(contrib.values())), 3), "features": feats, "contributions": contrib,
            "concept_table": table, "token_attributions": attrs}


def _diagram_record(i: int, box, crop_name: str, ref_name: str | None) -> dict:
    vlm = None
    if ref_name:
        vlm = {"components": {"score": 0.75, "reason": "Demo value: 3 of 4 boxes present."},
               "labels": {"score": 0.5, "reason": "Demo value: two labels missing."},
               "arrows": {"score": 1.0, "reason": "Demo value: arrows follow the reference."},
               "structure": {"score": 0.75, "reason": "Demo value: same top-to-bottom flow."},
               "missing_elements": ["label on box 3", "box 4"]}
    sim = {"clip": 0.86, "dino": 0.81}
    return {"idx": i, "bbox": [int(v) for v in box], "crop": crop_name, "ref": ref_name, "gradcam": None,
            "sim": sim, "sim_blend": 0.835, "sim_transformed": 0.71, "vlm": vlm,
            "score": 0.73 if vlm else 0.71, "needs_review": vlm is None,
            "reasons": [] if vlm else ["no reference diagram: similarity only"]}


def _write_page(pa, d: Path) -> tuple[str, list[int], float]:
    h, w = pa.pre.clean.shape[:2]
    scale = min(1.0, PAGE_MAX_W / w)
    page = cv2.resize(pa.pre.clean, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(d / "page_0.jpg"), page, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "page_0.jpg", [w, h], scale


def _score(text: dict, diagrams: list[dict], key: dict, cfg: Config) -> dict:
    total = float(key["total_marks"])
    dscore = float(np.mean([d["score"] for d in diagrams])) if diagrams else None
    wt, wd = (cfg.grading.weights.text, cfg.grading.weights.diagram) if dscore is not None else (1.0, 0.0)
    raw = (wt * text["score"] + wd * (dscore or 0.0)) * total
    step = cfg.grading.round_to
    return {"text_score": text["score"], "diagram_score": dscore, "weights": {"text": wt, "diagram": wd},
            "raw_marks": round(raw, 2), "marks": round(round(raw / step) * step, 2), "total_marks": total,
            "history_used": False, "history_expected": None}


def _justification(text: dict, score: dict, review: dict) -> str:
    tbl = text["concept_table"]
    got = [r["concept"] for r in tbl if r["status"] == "matched"]
    part = [r["concept"] for r in tbl if r["status"] == "partial"]
    miss = [r["concept"] for r in tbl if r["status"] == "missing"]
    s = f"Awarded {score['marks']} of {score['total_marks']:g} marks. "
    s += f"Key points covered: {', '.join(got) or 'none'}. "
    if part:
        s += f"Partly covered: {', '.join(part)}. "
    if miss:
        s += f"Missing: {', '.join(miss)}. "
    s += f"Text score {score['text_score']:.2f}"
    s += f", diagram score {score['diagram_score']:.2f}." if score["diagram_score"] is not None else "."
    if review["needed"]:
        s += " Teacher review needed: " + "; ".join(review["reasons"]) + "."
    return s


def _ocr_page(sheet_path, answer_key: dict, cfg: Config, out_dir: Path) -> tuple:
    rng = random.Random(str(sheet_path))
    pages = load_pages(sheet_path, cfg)
    pa = analyze_page(pages[0], cfg)
    page_name, page_size, _ = _write_page(pa, out_dir)
    texts = _demo_lines(len(pa.line_boxes), answer_key.get("reference_text", ""))
    lines = []
    for i, (box, crop) in enumerate(zip(pa.line_boxes, pa.line_crops)):
        _save_png(out_dir / f"line_{i:02d}.png", crop)
        lines.append(_line_record(i, box, f"line_{i:02d}.png", texts[i], rng, cfg, False))
    ref = None
    if answer_key.get("reference_diagram_path"):
        ref = "ref_diagram.png"
        img = cv2.imread(str(answer_key["reference_diagram_path"]))
        if img is None:
            raise BadImage("the reference diagram image is corrupt or unreadable")
        _save_png(out_dir / ref, img)
    diagrams = []
    for i, (box, crop) in enumerate(zip(pa.diagram_boxes, pa.diagram_crops)):
        _save_png(out_dir / f"diagram_{i}.png", crop)
        diagrams.append(_diagram_record(i, box, f"diagram_{i}.png", ref))
    notes = [f"demo pipeline reads page 1 only; pages 2-{len(pages)} were not evaluated"] if len(pages) > 1 else []
    return lines, diagrams, page_name, page_size, pa.pre.steps, notes


def _apply_edits(lines: list[dict], edited_lines: list[dict]) -> None:
    for e in edited_lines or []:
        ln = lines[int(e["idx"])]
        if e["text"] != ln["text"]:
            ln.setdefault("candidates", {})["teacher"] = e["text"]
            ln.update(text=e["text"], needs_review=False, review_reasons=["edited by teacher"], edited=True)


def evaluate_sheet(sheet_path, answer_key: dict, cfg: Config, student_id_anon: str | None = None,
                   topic: str | None = None, edited_lines: list[dict] | None = None,
                   out_dir: Path | None = None, prior: dict | None = None) -> dict:
    """Same signature/contract as src.pipeline.evaluate.evaluate_sheet (prior -> regrade, no re-OCR)."""
    out_dir = Path(out_dir)
    if prior is not None:
        lines, diagrams = prior["lines"], prior["diagrams"]
        page_name, page_size = prior["page_images"][0], prior.get("page_size")
        steps = prior.get("config_snapshot", {}).get("preprocess_steps", [])
        notes = prior.get("config_snapshot", {}).get("page_notes", [])
        _apply_edits(lines, edited_lines)
    else:
        lines, diagrams, page_name, page_size, steps, notes = _ocr_page(sheet_path, answer_key, cfg, out_dir)
    text = _text_grade(answer_key.get("reference_text", ""), " ".join(l["text"] for l in lines))
    score = _score(text, diagrams, answer_key, cfg)
    reasons = ["demo pipeline: OCR and scores are placeholders, not model output", *notes]
    reasons += [f"line {l['idx'] + 1}: {r}" for l in lines for r in l["review_reasons"] if r != "edited by teacher"]
    reasons += [f"diagram {d['idx'] + 1}: {r}" for d in diagrams for r in d["reasons"]]
    review = {"needed": True, "reasons": reasons}
    return {"sheet_id": (prior or {}).get("sheet_id") or Path(sheet_path).stem,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "page_images": [page_name], "page_size": page_size, "lines": lines, "diagrams": diagrams,
            "text_grade": text, "score": score, "history": None, "review": review,
            "justification": {"text": _justification(text, score, review), "engine": "template(demo)"},
            "override": None, "config_snapshot": {"ocr_engine": "demo", "preprocess_steps": steps, "page_notes": notes}}

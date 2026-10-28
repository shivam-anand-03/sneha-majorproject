"""End-to-end evaluation of one answer sheet -> result dict (contract used by the app and reports).

Stages: pages -> preprocess/layout/lines -> OCR (+confidence routing) -> text grading
-> diagram grading -> history (feedback / anomaly; marks only if use_history_in_score)
-> review flags -> XAI evidence + justification.
Optional modules degrade gracefully: if one fails its section is null and a review
reason is recorded, never a silent fake value.
"""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.common.config import Config, to_plain
from src.ocr.base import OCRLine
from src.ocr.router import recognize_lines
from src.pipeline.page import analyze_page
from src.preprocess.pdf import load_pages

log = logging.getLogger(__name__)


def _save(img: np.ndarray, out_dir: Path, name: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / name), img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return name


def _round_marks(x: float, cfg: Config) -> float:
    step = cfg.grading.round_to
    return float(np.round(x / step) * step) if step else float(x)


def _ocr_pages(pages, cfg, out_dir, flags) -> tuple[list[dict], list[dict], list[np.ndarray], list[str]]:
    lines, diagrams, diag_crops, page_names = [], [], [], []
    for p, img in enumerate(pages):
        pa = analyze_page(img, cfg, preprocess=flags.get("preprocess"), split_diagrams=flags.get("split_diagrams", True))
        page_names.append(_save(pa.pre.clean, out_dir, f"page_{p}.png"))
        ocr = recognize_lines(pa.line_crops, cfg) if pa.line_crops else []
        for b, crop, o in zip(pa.line_boxes, pa.line_crops, ocr):
            i = len(lines)
            lines.append({"idx": i, "page": p, "bbox": list(map(int, b)), "crop": _save(crop, out_dir, f"line_{i:03d}.png"),
                          **{k: v for k, v in o.to_dict().items() if k not in ("seconds",)}})
        for b, crop in zip(pa.diagram_boxes, pa.diagram_crops):
            i = len(diagrams)
            diagrams.append({"idx": i, "page": p, "bbox": list(map(int, b)), "crop": _save(crop, out_dir, f"diagram_{i}.png")})
            diag_crops.append(crop)
    return lines, diagrams, diag_crops, page_names


def _text_section(question, reference, answer, cfg, reasons) -> dict:
    from src.grading_text.scorer import grade_text

    tg = grade_text(question, reference, answer, cfg)
    sec = {"score": tg.score, "features": tg.features, "contributions": tg.contributions,
           "concept_table": tg.concept_table, "token_attributions": []}
    if cfg.xai.attribution == "none":
        return sec
    try:
        from src.xai.attribution import attribute_tokens

        sec["token_attributions"] = attribute_tokens(question, reference, answer, cfg)
    except Exception as e:
        reasons.append(f"token attribution unavailable: {type(e).__name__}")
    return sec


def _diagram_section(diagrams, diag_crops, ref_img, rubric, cfg, out_dir, reasons) -> float | None:
    if ref_img is None or not diag_crops:
        if ref_img is not None:
            reasons.append("answer key has a reference diagram but no diagram was detected on the sheet")
            return 0.0
        return None
    from src.grading_diagram.scorer import grade_diagram

    out_dir.joinpath("ref_diagram.png").parent.mkdir(parents=True, exist_ok=True)
    ref_name = _save(ref_img, out_dir, "ref_diagram.png")
    scores = []
    for d, crop in zip(diagrams, diag_crops):
        g = grade_diagram(crop, ref_img, rubric, cfg)
        d.update(g, ref=ref_name, gradcam=None)
        if cfg.xai.gradcam_target == "none":
            scores.append(float(g["score"]))
            continue
        try:
            from src.xai.gradcam import diagram_heatmap

            d["gradcam"] = _save(diagram_heatmap(crop, ref_img, cfg), out_dir, f"gradcam_{d['idx']}.png")
        except Exception as e:
            d["reasons"] = d.get("reasons", []) + [f"heatmap unavailable: {type(e).__name__}"]
        scores.append(float(g["score"]))
    return max(scores)  # best-matching detected region is the student's diagram


def _history_section(student_id, topic, raw_marks, total, text_tab, cfg, reasons) -> dict | None:
    if not student_id:
        return None
    try:
        from src.history.pipeline_hook import history_for_result

        return history_for_result(student_id, topic, raw_marks / total if total else 0.0, text_tab, cfg)
    except Exception as e:
        reasons.append(f"history unavailable: {type(e).__name__}: {str(e)[:80]}")
        return None


def evaluate_sheet(sheet_path: str | Path, answer_key: dict, cfg: Config, student_id_anon: str | None = None,
                   topic: str | None = None, edited_lines: list[dict] | None = None,
                   out_dir: str | Path | None = None, flags: dict | None = None,
                   prior: dict | None = None) -> dict:
    """answer_key: {question, reference_text, total_marks, rubric?, reference_diagram_path?}.
    edited_lines + prior: re-grade a previous result using teacher-edited OCR text (no re-OCR)."""
    flags = flags or {}
    rid = (prior or {}).get("result_id") or uuid.uuid4().hex[:12]
    out_dir = Path(out_dir or Path(cfg.project.output_dir) / "app_results" / rid)
    reasons: list[str] = []
    t0 = time.time()
    if prior is not None:
        lines, diagrams, page_names = prior["lines"], prior["diagrams"], prior["page_images"]
        diag_crops = [np.array(Image.open(out_dir / d["crop"]).convert("L")) for d in diagrams]
        for e in edited_lines or []:
            ln = lines[e["idx"]]
            if e["text"] != ln["text"]:
                ln.setdefault("candidates", {})["teacher"] = e["text"]
                ln["text"], ln["needs_review"] = e["text"], False
                ln["review_reasons"] = ["edited by teacher"]
    else:
        lines, diagrams, diag_crops, page_names = _ocr_pages(load_pages(sheet_path, cfg), cfg, out_dir, flags)
    return _grade(lines, diagrams, diag_crops, page_names, answer_key, cfg, student_id_anon, topic,
                  out_dir, rid, reasons, t0, Path(sheet_path).stem if sheet_path else rid)


def _grade(lines, diagrams, diag_crops, page_names, key, cfg, sid, topic, out_dir, rid, reasons, t0, sheet_id) -> dict:
    total = float(key["total_marks"])
    answer = key.get("oracle_answer") or " ".join(l["text"] for l in lines if l["text"].strip())
    text = _text_section(key.get("question", ""), key["reference_text"], answer, cfg, reasons)
    ref_img = None
    if key.get("reference_diagram_path"):
        ref_img = np.array(Image.open(key["reference_diagram_path"]).convert("RGB"))
    diag_score = _diagram_section(diagrams, diag_crops, ref_img, key.get("rubric", ""), cfg, out_dir, reasons)
    if diag_score is None:
        w = {"text": 1.0, "diagram": 0.0}
    else:
        w = {"text": cfg.grading.weights.text, "diagram": cfg.grading.weights.diagram}
    raw = (w["text"] * text["score"] + w["diagram"] * (diag_score or 0.0)) * total
    history = _history_section(sid, topic, raw, total, text["concept_table"], cfg, reasons)
    marks, used, expected = raw, False, None
    if history is not None:
        from src.history.apply import apply_history

        expected = history.get("expected")
        if expected is not None:
            ap = apply_history(raw / total, expected, cfg)
            marks, used = ap["marks"] * total, ap["history_used"]
    result = {
        "result_id": rid, "sheet_id": sheet_id, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "page_images": page_names, "lines": lines, "diagrams": diagrams, "text_grade": text,
        "score": {"text_score": text["score"], "diagram_score": diag_score, "weights": w,
                  "raw_marks": _round_marks(raw, cfg), "marks": _round_marks(marks, cfg), "total_marks": total,
                  "history_used": used, "history_expected": expected},
        "history": history, "override": None,
        "config_snapshot": {"ocr_engine": cfg.ocr.engine, "segmentation": cfg.segmentation.method,
                            "preprocess": to_plain(cfg.preprocess)["enabled"],
                            "use_history_in_score": cfg.history.use_history_in_score,
                            "cross_encoder": cfg.grading_text.cross_encoder.output_dir,
                            "embedder": cfg.grading_text.primary_embedder},
        "seconds": None,
    }
    _review_and_justify(result, cfg, reasons)
    result["seconds"] = round(time.time() - t0, 2)
    return result


def _review_and_justify(result: dict, cfg: Config, reasons: list[str]) -> None:
    try:
        from src.xai.review import needs_review

        needed, rr = needs_review(result, cfg)
    except Exception as e:
        needed, rr = True, [f"review module unavailable: {type(e).__name__}"]
    result["review"] = {"needed": bool(needed or reasons), "reasons": rr + reasons}
    try:
        from src.xai.justification import justify

        result["justification"] = justify(result, cfg)
    except Exception as e:
        result["justification"] = {"text": "", "engine": f"unavailable: {type(e).__name__}"}


def ocr_line_from_dict(d: dict) -> OCRLine:
    return OCRLine(**{k: d[k] for k in OCRLine.__dataclass_fields__ if k in d})

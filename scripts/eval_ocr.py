"""Stage 3: OCR comparison (CER/WER + seconds/line) on the same held-out lines.

Sets
  iam      : IAM test-split line images (real handwriting, writer-disjoint from TrOCR training)
  sheets   : synthetic sheets end to end (segmentation -> OCR), preprocessed vs raw page,
             each GT line matched to a predicted line by IoU; unmatched GT counts as empty output

python -m scripts.eval_ocr --engines trocr paddle vlm [gemini] --n-iam 200 --n-vlm 60
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from PIL import Image

from src.common.config import load_config, resolve, seed_everything
from src.common.gpu import free_memory
from src.common.tracking import log_experiment, save_json
from src.correction.byt5 import correction_status
from src.data.iam import load_iam_lines
from src.data.synthetic import load_manifest
from src.ocr.metrics import cer_wer
from src.ocr.router import _ENGINES, get_engine
from src.pipeline.page import analyze_page
from src.segmentation.metrics import match_boxes


def run_engine(name: str, crops: list[np.ndarray], cfg) -> tuple[list[str], float, list]:
    eng = get_engine(name, cfg)
    if hasattr(eng, "available") and not eng.available():
        raise RuntimeError(f"{name} unavailable (opt-in/key missing)")
    t0 = time.time()
    lines = eng.recognize(crops)
    return [l.text for l in lines], (time.time() - t0) / max(len(crops), 1), lines


def sheet_pairs(cfg, sheets: list[dict], preprocess: bool) -> tuple[list[np.ndarray], list[str], list[str]]:
    """Crops of predicted lines matched to GT text; returns crops, refs and GT texts with no matched line."""
    crops, refs, missed = [], [], []
    for s in sheets:
        img = np.array(Image.open(s["image_path"]).convert("RGB"))
        pa = analyze_page(img, cfg, preprocess=preprocess)
        gt = [l for l in s["lines"] if not l.get("header")]
        m = match_boxes(pa.line_boxes, [tuple(l["bbox"]) for l in gt], cfg.segmentation.eval.iou_threshold)
        got = {j: i for i, j, _ in m}
        for j, l in enumerate(gt):
            if j in got:
                crops.append(pa.line_crops[got[j]])
                refs.append(l["text"])
            else:
                missed.append(l["text"])
    return crops, refs, missed


def eval_set(cfg, label: str, crops, refs, engines, n_vlm, results, missed: list[str] | None = None) -> None:
    missed = missed or []
    for name in engines:
        n = n_vlm if name in ("vlm", "gemini") else len(crops)
        try:
            hyps, spl, lines = run_engine(name, crops[:n], cfg)
        except Exception as e:  # report, keep going
            results[f"{label}|{name}"] = {"error": f"{type(e).__name__}: {str(e)[:150]}"}
            continue
        m = cer_wer(refs[:n], hyps)  # matched lines only
        if missed and n == len(crops):
            # end to end: GT lines that segmentation missed count as empty transcriptions
            e2e = cer_wer(refs + missed, hyps + [""] * len(missed))
            m.update(cer_e2e=e2e["cer"], wer_e2e=e2e["wer"])
        confs = [l.confidence for l in lines if l.confidence is not None]
        results[f"{label}|{name}"] = {**m, "sec_per_line": round(spl, 3), "seg_missed_lines": len(missed),
                                      "mean_conf": float(np.mean(confs)) if confs else None,
                                      "review_rate": float(np.mean([l.needs_review for l in lines]))}
        print(label, name, results[f"{label}|{name}"], flush=True)
        if name in ("vlm",):
            _ENGINES.pop(name, None)
            free_memory()


def confidence_calibration(cfg, crops, refs) -> dict:
    """Does TrOCR line confidence predict errors? CER for low- vs high-confidence lines."""
    _, _, lines = run_engine("trocr", crops, cfg)
    thr = cfg.ocr.confidence.line_low_threshold
    lo = [(r, l.text) for r, l in zip(refs, lines) if l.confidence < thr]
    hi = [(r, l.text) for r, l in zip(refs, lines) if l.confidence >= thr]
    out = {"threshold": thr, "n_low": len(lo), "n_high": len(hi)}
    if lo:
        out["cer_low_conf"] = cer_wer(*map(list, zip(*lo)))["cer"]
    if hi:
        out["cer_high_conf"] = cer_wer(*map(list, zip(*hi)))["cer"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", nargs="+", default=["trocr", "paddle", "vlm"])
    ap.add_argument("--n-iam", type=int, default=200)
    ap.add_argument("--n-vlm", type=int, default=60)
    ap.add_argument("--n-sheets", type=int, default=30)
    args = ap.parse_args()
    cfg = load_config()
    seed_everything(cfg.project.seed)
    results: dict[str, dict] = {}
    iam = load_iam_lines("test", limit=args.n_iam, seed=cfg.project.seed + 1)
    crops = [np.array(im.convert("L")) for im, _ in iam]
    refs = [t for _, t in iam]
    eval_set(cfg, "iam", crops, refs, args.engines, args.n_vlm, results)
    results["iam|trocr_confidence"] = confidence_calibration(cfg, crops, refs)

    sheets = [s for s in load_manifest(resolve(cfg, cfg.paths.synthetic) / "manifest.json")][: args.n_sheets]
    for pre in (True, False):
        c, r, missed = sheet_pairs(cfg, sheets, pre)
        eval_set(cfg, f"sheets_{'pre' if pre else 'raw'}", c, r, ["trocr"], args.n_vlm, results, missed)
        results[f"sheets_{'pre' if pre else 'raw'}|trocr"]["gt_lines"] = len(r) + len(missed)
    results["correction"] = correction_status(cfg)
    table = pd.DataFrame(results).T
    print(table.to_string())
    out = resolve(cfg, cfg.project.output_dir)
    table.to_csv(out / "metrics_ocr.csv")
    save_json(results, out / "metrics_ocr.json")
    log_experiment(cfg, "ocr_comparison", results, config_sections=["ocr", "preprocess", "correction"])


if __name__ == "__main__":
    main()

"""End-to-end agreement with teacher marks on synthetic answer sheets + ablations.

Teacher marks per sheet = (w_t * Mohler human score + w_d * rule-based diagram mark) * total,
text-only sheets use the human score alone (same weighting policy as the pipeline).

Variants: full | no_preprocess | no_split (no text/diagram separation) | oracle_ocr
(ground-truth transcription, isolates OCR error). History on/off is in scripts/eval_history.py.

python -m scripts.eval_e2e [--limit N] [--vlm]
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from src.common.config import load_config, resolve, seed_everything
from src.common.tracking import log_experiment, save_json
from src.data.synthetic import load_manifest
from src.grading_text.metrics import grading_metrics
from src.pipeline.evaluate import evaluate_sheet

TOTAL = 10.0
VARIANTS = {
    "full": {},
    "no_preprocess": {"preprocess": False},
    "no_split": {"split_diagrams": False},
    "oracle_ocr": {"oracle": True},
}


def teacher_marks(s: dict, cfg) -> float:
    if s.get("diagram"):
        w = cfg.grading.weights
        return (w.text * s["teacher_text_score"] + w.diagram * s["diagram"]["mark"]) * TOTAL
    return s["teacher_text_score"] * TOTAL


def run_variant(name: str, flags: dict, sheets: list[dict], cfg) -> pd.DataFrame:
    rows = []
    for s in sheets:
        key = {"question": s["question"], "reference_text": s["reference"], "total_marks": TOTAL,
               "rubric": "", "reference_diagram_path": s["diagram"]["ref_path"] if s.get("diagram") else None}
        if flags.get("oracle"):
            key["oracle_answer"] = s["answer_text"]
        t0 = time.time()
        r = evaluate_sheet(s["image_path"], key, cfg, out_dir=resolve(cfg, cfg.project.output_dir) / "e2e" / name / s["sheet_id"],
                           flags={k: v for k, v in flags.items() if k != "oracle"})
        rows.append({"sheet": s["sheet_id"], "pred": r["score"]["marks"], "true": teacher_marks(s, cfg),
                     "text_score": r["score"]["text_score"], "diagram_score": r["score"]["diagram_score"],
                     "true_text": s["teacher_text_score"], "true_diag": s["diagram"]["mark"] if s.get("diagram") else None,
                     "review": r["review"]["needed"], "sec": time.time() - t0,
                     **{f"deg_{k}": v for k, v in s["degradations"].items()}})
        print(name, rows[-1]["sheet"], round(rows[-1]["pred"], 2), round(rows[-1]["true"], 2), flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--vlm", action="store_true", help="include VLM diagram rubric (slow)")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = ap.parse_args()
    cfg = load_config(overrides={"xai": {"attribution": "none", "gradcam_target": "none",
                                         "justification": {"engine": "template"}},
                                 "grading_diagram": {"vlm_rubric": {"enabled": args.vlm}},
                                 "ocr": {"confidence": {"use_vlm_fallback": False}}})
    seed_everything(cfg.project.seed)
    sheets = [s for s in load_manifest(resolve(cfg, cfg.paths.synthetic) / "manifest.json") if s["kind"] == "answer"]
    sheets = sheets[: args.limit]
    results, frames = {}, {}
    for name in args.variants:
        df = run_variant(name, VARIANTS[name], sheets, cfg)
        frames[name] = df
        results[name] = {**grading_metrics(df.pred, df.true, TOTAL, step=0.5, tol=1.0),
                         "review_rate": float(df.review.mean()), "sec_per_sheet": float(df.sec.mean())}
        for cond, sub in [("ruled", df[df.deg_ruled]), ("plain", df[~df.deg_ruled]),
                          ("shadow", df[df.deg_shadow]), ("skew>3", df[df.deg_skew_deg.abs() > 3])]:
            if len(sub) > 3:
                results[name][f"mae_{cond}"] = float((sub.pred - sub.true).abs().mean())
        df.to_csv(resolve(cfg, cfg.project.output_dir) / f"e2e_{name}.csv", index=False)
    table = pd.DataFrame(results).T
    print(table.round(3).to_string())
    out = resolve(cfg, cfg.project.output_dir)
    save_json(results, out / "metrics_e2e.json")
    log_experiment(cfg, "e2e", results, extra={"vlm": args.vlm, "n_sheets": len(sheets)})


if __name__ == "__main__":
    main()

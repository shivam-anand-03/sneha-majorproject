"""Stage 2 evaluation: line recall/precision and diagram detection on synthetic sheets.

python -m scripts.eval_segmentation [--methods projection paddle] [--debug 3]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from PIL import Image

from src.common.config import load_config, resolve
from src.common.tracking import log_experiment, save_json
from src.data.synthetic import load_manifest
from src.pipeline.page import analyze_page, save_debug
from src.segmentation.metrics import iou, line_pr


def evaluate(cfg, sheets, method: str, preprocess: bool, split: bool, debug: int = 0) -> pd.DataFrame:
    thr = cfg.segmentation.eval.iou_threshold
    rows = []
    for i, s in enumerate(sheets):
        img = np.array(Image.open(s["image_path"]).convert("RGB"))
        pa = analyze_page(img, cfg, preprocess=preprocess, split_diagrams=split, seg_method=method)
        gt = [tuple(l["bbox"]) for l in s["lines"]]
        m = line_pr(pa.line_boxes, gt, thr)
        d_gt = s.get("diagram")
        d_hit = None
        if d_gt:
            d_hit = any(iou(b, tuple(d_gt["bbox"])) >= thr for b in pa.diagram_boxes)
        rows.append({"sheet": s["sheet_id"], "kind": s["kind"], **m, "has_diagram": bool(d_gt),
                     "diagram_found": d_hit, "false_diagrams": len(pa.diagram_boxes) - int(bool(d_hit)),
                     "skew_true": s["degradations"]["skew_deg"], "skew_est": pa.pre.skew_angle,
                     "ruled": s["degradations"]["ruled"], "shadow": s["degradations"]["shadow"]})
        if i < debug:
            save_debug(pa, resolve(cfg, cfg.project.output_dir) / "debug_crops", f"{method}_{s['sheet_id']}")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    tp, npred, ngt = df.tp.sum(), df.n_pred.sum(), df.n_gt.sum()
    d = df[df.has_diagram]
    return {"line_recall": tp / ngt, "line_precision": tp / max(npred, 1), "sheets": len(df),
            "diagram_recall": float(d.diagram_found.mean()) if len(d) else None,
            "false_diagrams_per_sheet": float(df.false_diagrams.clip(lower=0).mean()),
            "skew_abs_err_deg": float((df.skew_true - df.skew_est).abs().mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["projection"])
    ap.add_argument("--debug", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config()
    sheets = load_manifest(resolve(cfg, cfg.paths.synthetic) / "manifest.json")[: args.limit]
    results = {}
    for method in args.methods:
        for pre in (True, False):
            df = evaluate(cfg, sheets, method, pre, True, args.debug if pre else 0)
            key = f"{method}|{'pre' if pre else 'raw'}"
            results[key] = summarize(df)
            for cond, sub in [("ruled", df[df.ruled]), ("plain", df[~df.ruled]), ("shadow", df[df.shadow]),
                              ("skew>3", df[df.skew_true.abs() > 3])]:
                if len(sub):
                    results[key][f"recall_{cond}"] = sub.tp.sum() / sub.n_gt.sum()
            df.to_csv(resolve(cfg, cfg.project.output_dir) / f"seg_{method}_{'pre' if pre else 'raw'}.csv", index=False)
    table = pd.DataFrame(results).T
    print(table.round(3).to_string())
    out = resolve(cfg, cfg.project.output_dir)
    save_json(results, out / "metrics_segmentation.json")
    log_experiment(cfg, "segmentation", results, config_sections=["preprocess", "segmentation", "layout"])


if __name__ == "__main__":
    main()

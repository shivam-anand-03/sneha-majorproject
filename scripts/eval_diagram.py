"""Stage 5 evaluation: diagram extraction + grading ablation (CLIP / +DINOv2 / +SigLIP / +VLM).

python -m scripts.eval_diagram [--n-synth 150] [--vlm-n 40] [--device cpu]
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import pearsonr, spearmanr

from src.common.config import load_config, resolve, seed_everything
from src.common.tracking import log_experiment, save_json
from src.data.synthetic import font_paths, load_manifest
from src.grading_diagram import embed_sim, synth, vlm_rubric
from src.pipeline.page import analyze_page
from src.segmentation.metrics import iou

VARIANTS = {"clip": ["clip"], "dino": ["dino"], "clip+dino": ["clip", "dino"],
            "clip+dino+siglip": ["clip", "dino", "siglip"]}


def sheet_pairs(cfg) -> tuple[list[dict], list[float]]:
    """Student diagram crops extracted by the real pipeline from synthetic sheets."""
    pairs, ious = [], []
    for s in load_manifest(resolve(cfg, cfg.paths.synthetic) / "manifest.json"):
        if not s.get("diagram"):
            continue
        pa = analyze_page(np.array(Image.open(s["image_path"]).convert("RGB")), cfg)
        gt = tuple(s["diagram"]["bbox"])
        scores = [iou(b, gt) for b in pa.diagram_boxes]
        ious.append(max(scores) if scores else 0.0)
        if not scores:
            continue  # extraction failure: counted in IoU, nothing to grade
        crop = pa.diagram_crops[int(np.argmax(scores))]
        pairs.append({"source": "sheet", "student": crop, "ref": np.array(Image.open(s["diagram"]["ref_path"]).convert("RGB")),
                      "mark": s["diagram"]["mark"], "defects": s["diagram"]["defects"]})
    return pairs, ious


def synth_pairs(cfg, n: int) -> list[dict]:
    rng = random.Random(cfg.project.seed + 5)
    fonts = font_paths(cfg)
    ref_font = str(resolve(cfg, cfg.paths.fonts) / "PatrickHand.ttf")
    out = []
    for _ in range(n):
        ref = synth.reference_spec(rng)
        stu = synth.student_variant(ref, rng, n_defects=rng.choice([0, 1, 1, 2, 3]))
        size = (rng.randint(800, 1100), rng.randint(380, 520))
        out.append({"source": "synth", "mark": stu.mark, "defects": stu.defects,
                    "student": synth.render(stu, rng.choice(fonts), size, rng),
                    "ref": synth.render(ref, ref_font, size, random.Random(0))})
    return out


def corr(pred, mark) -> dict:
    p, t = np.asarray(pred, float), np.asarray(mark, float)
    ok = p.std() > 0 and t.std() > 0
    return {"pearson": float(pearsonr(p, t)[0]) if ok else float("nan"),
            "spearman": float(spearmanr(p, t)[0]) if ok else float("nan"),
            "mae": float(np.mean(np.abs(p - t))), "n": int(len(p))}


def embedding_scores(pairs: list[dict], cfg) -> pd.DataFrame:
    df = pd.DataFrame({"mark": [p["mark"] for p in pairs], "source": [p["source"] for p in pairs],
                       "defects": [",".join(p["defects"]) or "none" for p in pairs]})
    for kind in ["clip", "dino", "siglip"]:
        es = embed_sim.embed(kind, [p["student"] for p in pairs], cfg)
        er = embed_sim.embed(kind, [p["ref"] for p in pairs], cfg)
        df[kind] = np.clip((es * er).sum(1), 0, 1)
    g = cfg.grading_diagram
    for name, kinds in VARIANTS.items():
        w = {k: 1.0 for k in kinds} if name != "clip+dino" else dict(g.embed_weights)
        raw = df[kinds].apply(lambda r: embed_sim.blend(r.to_dict(), w), axis=1).to_numpy()
        df[f"{name}|raw"] = raw
        df[f"{name}|T"] = embed_sim.paper_transform(raw, g.transform.k, g.transform.c)
        df[f"{name}|T01"] = embed_sim.transform01(raw, g.transform.k, g.transform.c)
    return df


def vlm_scores(pairs: list[dict], idx: list[int], cfg) -> dict[int, float]:
    out = {}
    for i in idx:
        res, why = vlm_rubric.rubric_check(pairs[i]["student"], pairs[i]["ref"], "", cfg)
        if res is None:
            print("VLM skipped:", why)
            break
        out[i] = res["score"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-synth", type=int, default=150)
    ap.add_argument("--vlm-n", type=int, default=40)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = load_config(overrides={"project": {"device": args.device}} if args.device else None)
    seed_everything(cfg.project.seed)
    sp, ious = sheet_pairs(cfg)
    pairs = sp + synth_pairs(cfg, args.n_synth)
    df = embedding_scores(pairs, cfg)
    results = {"extraction": {"n_sheets": len(ious), "mean_iou": float(np.mean(ious)),
                              "recall@0.5": float(np.mean(np.array(ious) >= 0.5))}}
    score_cols = [c for c in df.columns if "|" in c]
    results["all"] = {c: corr(df[c], df.mark) for c in score_cols}
    sheet = df[df.source == "sheet"]
    t01_cols = [c for c in score_cols if c.endswith("|T01")]
    results["sheet_only"] = {c: corr(sheet[c], sheet.mark) for c in t01_cols}
    single = df[~df.defects.str.contains(",")]
    results["per_defect_mean"] = single.groupby("defects")[["mark", "clip|raw", "clip+dino|raw", "clip+dino|T01"]].mean().round(3).to_dict()
    results.update(vlm_ablation(pairs, df, cfg, args.vlm_n))
    show = pd.DataFrame(results["all"]).T.round(3)
    print(show.to_string())
    print(pd.DataFrame(results["per_defect_mean"]).to_string())
    out = resolve(cfg, cfg.project.output_dir)
    df.to_csv(out / "diagram_scores.csv", index=False)
    save_json(results, out / "metrics_diagram.json")
    log_experiment(cfg, "diagram_grading", results, config_sections=["grading_diagram", "layout"])


def vlm_ablation(pairs, df, cfg, n: int) -> dict:
    if n <= 0:
        return {"vlm": "skipped (--vlm-n 0)"}
    rng = np.random.default_rng(cfg.project.seed)
    idx = sorted(rng.choice(len(pairs), size=min(n, len(pairs)), replace=False).tolist())
    vs = vlm_scores(pairs, idx, cfg)
    if not vs:
        return {"vlm": "skipped (VLM unavailable)"}
    sub = df.loc[list(vs)].copy()
    sub["vlm"] = [vs[i] for i in vs]
    w = cfg.grading_diagram.blend
    sub["clip+dino+vlm"] = (w.w_sim * sub["clip+dino|T01"] + w.w_vlm * sub.vlm) / (w.w_sim + w.w_vlm)
    cols = ["clip|T01", "clip+dino|T01", "vlm", "clip+dino+vlm"]
    sub.to_csv(resolve(cfg, cfg.project.output_dir) / "diagram_vlm_subset.csv", index=False)
    return {"vlm_subset": {c: corr(sub[c], sub.mark) for c in cols}}


if __name__ == "__main__":
    main()

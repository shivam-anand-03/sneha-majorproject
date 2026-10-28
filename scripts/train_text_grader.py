"""Stage 4: fine-tune the cross-encoder, fit combiner weights, evaluate text grading.

python -m scripts.train_text_grader [--skip-train] [--max-train N]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.common.config import load_config, resolve, seed_everything
from src.common.gpu import free_memory
from src.common.tracking import log_experiment, save_json
from src.data.asag import load_asag
from src.grading_text import combiner, concepts, cross_encoder, embed
from src.grading_text.metrics import grading_metrics
from src.grading_text.robustness import build_probes


def compute_features(df: pd.DataFrame, cfg, with_concepts: bool = True) -> dict[str, np.ndarray]:
    feats = {
        "embed_sim": embed.pair_similarity(df.reference.tolist(), df.answer.tolist(), cfg),
        "cross_encoder": cross_encoder.predict(df.question.tolist(), df.reference.tolist(), df.answer.tolist(), cfg),
    }
    if with_concepts:
        cache: dict[str, list[str]] = {}
        cov = []
        for r, a in zip(df.reference, df.answer):
            if r not in cache:
                cache[r] = concepts.extract_concepts(r, cfg)
            cov.append(concepts.coverage(concepts.match_concepts(cache[r], a, cfg)))
        feats["concept_coverage"] = np.asarray(cov)
    return feats


def eval_block(df: pd.DataFrame, pred01: np.ndarray, cfg) -> dict:
    mm = df.max_marks.to_numpy()
    step = 0.5 if mm.max() > 1 else 0.5 * mm.max()
    return grading_metrics(pred01 * mm, df.score.to_numpy() * mm, mm, step=step,
                           tol=cfg.grading_text.within_tolerance * (1 if mm.max() > 1 else 0.5))


def robustness_probes(moh: pd.DataFrame, cfg, w: dict) -> pd.DataFrame:
    """Mean normalised scores per probe type on Mohler test questions."""
    probes = build_probes(moh, lambda r: concepts.extract_concepts(r, cfg), cfg.project.seed)
    pf = compute_features(probes, cfg)
    probes["combined"] = combiner.predict(pf, cfg, w)
    for k in ("embed_sim", "cross_encoder", "concept_coverage"):
        probes[k] = pf[k]
    rob = probes.groupby("probe")[["embed_sim", "cross_encoder", "concept_coverage", "combined"]].mean()
    print("\nRobustness (mean normalised score; stuffing/salad should be low):\n", rob.round(3))
    return rob


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--max-train", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config()
    seed_everything(cfg.project.seed)
    df = load_asag(cfg)
    print(df.groupby(["dataset", "split"]).size().unstack(fill_value=0))
    train, val = df[df.split == "train"], df[df.split == "val"]
    if args.max_train:
        train = train.sample(n=min(args.max_train, len(train)), random_state=cfg.project.seed)
    if not args.skip_train:
        t0 = time.time()
        cross_encoder.train_cross_encoder(train, val, cfg)
        print(f"cross-encoder trained in {time.time() - t0:.0f}s")
        free_memory()

    # combiner is fitted on VALIDATION data the cross-encoder never trained on
    vf = compute_features(val, cfg)
    w = combiner.fit(vf, val.score.to_numpy(), cfg)
    print("combiner weights:", w)

    results: dict[str, dict] = {}
    tests = {"mohler_test": df[(df.dataset == "mohler") & (df.split == "test")],
             "scientsbank_ua": df[(df.dataset == "scientsbank") & (df.split == "test")],
             "scientsbank_uq": df[(df.dataset == "scientsbank") & (df.split == "test_uq")],
             "beetle_ua": df[(df.dataset == "beetle") & (df.split == "test")]}
    for name, t in tests.items():
        if not len(t):
            continue
        f = compute_features(t, cfg)
        # each single signal is linearly calibrated on val so all variants are on the same scale
        for feat in ["embed_sim", "cross_encoder", "concept_coverage"]:
            from sklearn.linear_model import LinearRegression

            lr = LinearRegression().fit(vf[feat].reshape(-1, 1), val.score)
            results[f"{name}|{feat}_only"] = eval_block(t, np.clip(lr.predict(f[feat].reshape(-1, 1)), 0, 1), cfg)
        results[f"{name}|combined"] = eval_block(t, combiner.predict(f, cfg, w), cfg)
        for emb_name in cfg.grading_text.embedders:
            sim = embed.pair_similarity(t.reference.tolist(), t.answer.tolist(), cfg, emb_name)
            results[f"{name}|raw_cosine:{emb_name.split('/')[-1]}"] = eval_block(t, sim, cfg)

    rob = robustness_probes(tests["mohler_test"], cfg, w)

    table = pd.DataFrame(results).T
    print("\n", table.round(3).to_string())
    out = resolve(cfg, cfg.project.output_dir)
    table.to_csv(out / "metrics_text_grading.csv")
    rob.to_csv(out / "robustness_text.csv")
    save_json({"results": results, "robustness": rob.to_dict(), "combiner": w}, out / "metrics_text_grading.json")
    log_experiment(cfg, "text_grading", {"results": results, "robustness": rob.to_dict()},
                   config_sections=["grading_text"])


if __name__ == "__main__":
    main()

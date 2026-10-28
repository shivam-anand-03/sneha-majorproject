"""Stage 7 evaluation: history model quality, use_history_in_score ablation, fairness check.

python -m scripts.eval_history
All histories are SYNTHETIC (no real learning-history data exists); see README.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from src.common.config import load_config, resolve, seed_everything
from src.common.tracking import log_experiment, save_json
from src.data.asag import load_mohler
from src.grading_text.metrics import grading_metrics
from src.history.apply import apply_history
from src.history.model import build_training_frame, expected_score, fit_history_model
from src.history.synthetic import make_history

BLENDS = [0.05, 0.1, 0.2, 0.3]


def eval_predictive(cfg) -> dict:
    """Temporal split per student: last 25% of each student's attempts are held out."""
    hist = make_history([f"H{i:04d}" for i in range(300)], cfg.project.seed)
    fr = build_training_frame(hist, cfg)
    fr["rank"] = fr.groupby("student").cumcount()
    fr["n"] = fr.groupby("student").student.transform("size")
    test_mask = fr["rank"] >= (fr["n"] * 0.75).astype(int)
    tr, te = fr[~test_mask], fr[test_mask]
    out = {}
    for kind in ["mean", "ridge", "lgbm"]:
        hm = fit_history_model(tr, cfg, kind=kind)
        p = hm.predict_frame(te)
        out[kind] = {"mae": float(np.abs(p - te.target).mean()), "pearson": float(pearsonr(p, te.target)[0]),
                     "n_test": int(len(te))}
    return out


def model_marks_for(test: pd.DataFrame, cfg) -> tuple[np.ndarray, str]:
    """Real text-grader predictions if trained, else teacher score + noise (clearly labelled)."""
    w, ce = resolve(cfg, cfg.grading_text.combiner.weights_path), resolve(cfg, cfg.grading_text.cross_encoder.output_dir)
    if w.exists() and ce.exists():
        from scripts.train_text_grader import compute_features
        from src.grading_text import combiner

        cpu = load_config(overrides={"project": {"device": "cpu"}})
        return combiner.predict(compute_features(test, cpu), cpu), "text_grader"
    rng = np.random.default_rng(cfg.project.seed)
    return np.clip(test.score.to_numpy() + rng.normal(0, 0.15, len(test)), 0, 1), "teacher_plus_noise(sd=0.15)"


def history_expectations(test: pd.DataFrame, cfg) -> np.ndarray:
    """Synthetic history anchored to each test student's true ability, then LightGBM expectations."""
    test = test.assign(topic=test.qid.str.split(".").str[0])
    ability = test.groupby("group").score.mean().to_dict()
    hist = make_history(sorted(ability), cfg.project.seed + 7, ability=ability)
    hm = fit_history_model(build_training_frame(hist, cfg), cfg)
    cache: dict[tuple, float] = {}
    exp = []
    for g, t in zip(test.group, test.topic):
        if (g, t) not in cache:
            cache[(g, t)] = expected_score(hm, hist, g, t, cfg)
        exp.append(cache[(g, t)])
    return np.asarray(exp, float)


def _blend_cfg(b: float | None):
    return load_config(overrides={"history": {"use_history_in_score": b is not None, "history_blend": b or 0.0}})


def ablation_and_fairness(cfg) -> tuple[dict, dict, str]:
    moh = load_mohler(cfg)
    test = moh[moh.split == "test"].reset_index(drop=True)
    mm = test.max_marks.to_numpy()
    pred, source = model_marks_for(test, cfg)
    exp = history_expectations(test, cfg)
    ability = test.groupby("group").score.transform("mean")
    tercile = pd.qcut(ability.rank(method="first"), 3, labels=["low", "mid", "high"])
    atypical = (test.score - ability).abs() > 0.25
    abl, fair = {}, {}
    for b in [None] + BLENDS:
        c = _blend_cfg(b)
        final = np.array([apply_history(p, e, c)["marks"] * m for p, e, m in zip(pred, exp, mm)])
        key = "history_off" if b is None else f"history_on_b{b}"
        abl[key] = grading_metrics(final, test.score * mm, mm, step=0.5, tol=0.5)
        err = final - test.score.to_numpy() * mm
        fair[key] = {f"{t}_signed_err": float(err[tercile == t].mean()) for t in ["low", "mid", "high"]}
        fair[key].update({f"{t}_mae": float(np.abs(err[tercile == t]).mean()) for t in ["low", "mid", "high"]})
        fair[key]["atypical_answers_mae"] = float(np.abs(err[atypical]).mean())
        fair[key]["n_atypical"] = int(atypical.sum())
    return abl, fair, source


def main() -> None:
    cfg = load_config(overrides={"project": {"device": "cpu"}})
    seed_everything(cfg.project.seed)
    pred = eval_predictive(cfg)
    print("History model (held-out later attempts):\n", pd.DataFrame(pred).T.round(3))
    abl, fair, source = ablation_and_fairness(cfg)
    print(f"\nAblation on Mohler test (model marks source: {source}):\n", pd.DataFrame(abl).T.round(3).to_string())
    print("\nFairness (error in marks by ability tercile / atypical answers):\n", pd.DataFrame(fair).T.round(3).to_string())
    res = {"predictive": pred, "ablation": abl, "fairness": fair, "model_marks_source": source,
           "note": "histories are synthetic; ability anchored to observed Mohler scores"}
    save_json(res, resolve(cfg, cfg.project.output_dir) / "metrics_history.json")
    log_experiment(cfg, "history", res, config_sections=["history"])


if __name__ == "__main__":
    main()

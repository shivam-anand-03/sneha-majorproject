"""Stage 6 evaluation: attribution faithfulness (deletion test) + justification provenance.

python -m scripts.eval_xai [--n 100] [--n-just 10] [--device cpu]

The teacher usefulness study cannot be run without teachers; a blank review form is written
to outputs/teacher_review_form.csv with template explanations to rate.
"""
from __future__ import annotations

import argparse
import random
import time

import pandas as pd

from src.common.config import load_config, resolve, seed_everything
from src.common.tracking import log_experiment, save_json
from src.data.asag import load_mohler
from src.grading_text.cross_encoder import first_text, model_path
from src.xai.attribution import integrated_gradients
from src.xai.faithfulness import deletion_test, summarize
from src.xai.justification import evidence_from_result, justify, template_justification, unsupported_numbers

FALLBACK_CE = "cross-encoder/stsb-roberta-base"


def load_ce(cfg):
    import torch
    from sentence_transformers.cross_encoder import CrossEncoder

    from src.common.gpu import get_device

    path = model_path(cfg)
    trained = (path / "config.json").exists()
    name = str(path) if trained else FALLBACK_CE
    ce = CrossEncoder(name, device=get_device(cfg.project.device), activation_fn=torch.nn.Sigmoid(),
                      max_length=cfg.grading_text.cross_encoder.max_length)
    return ce, name, trained


def faithfulness(ce, items: pd.DataFrame, cfg) -> tuple[dict, list[dict]]:
    rng = random.Random(cfg.project.seed)
    ks = list(cfg.xai.faithfulness_delete_k)

    def score(first, ans):
        return float(ce.predict([[first, ans]], show_progress_bar=False)[0])

    res, examples = [], []
    for _, it in items.iterrows():
        first = first_text(it.question, it.reference)
        words = integrated_gradients(ce, first, it.answer, cfg)
        res.append(deletion_test(score, first, it.answer, words, ks, rng))
        examples.append({"qid": it.qid, "answer": it.answer, "words": words, "score": res[-1]["base"]})
    return summarize(res, ks), examples


def _fake_result(ex: dict, it) -> dict:
    """Minimal result dict (text only) for justification evaluation."""
    from src.grading_text import concepts

    table = concepts.match_concepts(concepts.extract_concepts(it.reference, CFG), it.answer, CFG)
    s = round(ex["score"], 2)
    return {"lines": [], "diagrams": [], "history": None,
            "text_grade": {"features": {"cross_encoder": s, "concept_coverage": round(concepts.coverage(table), 2)},
                           "concept_table": table, "token_attributions": ex["words"]},
            "score": {"text_score": s, "diagram_score": None, "weights": {"text": 1.0, "diagram": 0.0},
                      "raw_marks": round(s * 5 * 2) / 2, "marks": round(s * 5 * 2) / 2, "total_marks": 5.0,
                      "history_used": False, "history_expected": None},
            "review": {"needed": False, "reasons": []}}


def justification_eval(examples, items, cfg, n: int) -> tuple[dict, list[dict]]:
    rows = []
    for ex, (_, it) in list(zip(examples, items.iterrows()))[:n]:
        r = _fake_result(ex, it)
        t0 = time.time()
        out = justify(r, cfg) if n else {}
        rows.append({"qid": it.qid, "answer": it.answer, "engine": out.get("engine"),
                     "rejected": out.get("rejected_reason"), "text": out.get("text"),
                     "template": template_justification(evidence_from_result(r)),
                     "template_unsupported": unsupported_numbers(template_justification(evidence_from_result(r)),
                                                                 evidence_from_result(r)),
                     "seconds": time.time() - t0})
    llm_rows = [x for x in rows if x["engine"] != "template" or (x["rejected"] or "").startswith("unsupported")]
    rej = [x for x in llm_rows if (x["rejected"] or "").startswith("unsupported")]
    return {"n": len(rows), "llm_attempts": len(llm_rows),
            "provenance_rejection_rate": len(rej) / len(llm_rows) if llm_rows else None,
            "template_with_unsupported_numbers": sum(bool(x["template_unsupported"]) for x in rows)}, rows


def main() -> None:
    global CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--n-just", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    cfg = CFG = load_config(overrides={"project": {"device": args.device}})
    seed_everything(cfg.project.seed)
    moh = load_mohler(cfg)
    items = moh[moh.split == "test"].sample(n=args.n, random_state=cfg.project.seed)
    ce, ce_name, trained = load_ce(cfg)
    faith, examples = faithfulness(ce, items, cfg)
    print("faithfulness:", faith)
    just, rows = justification_eval(examples, items, cfg, args.n_just)
    print("justification:", just)
    out = resolve(cfg, cfg.project.output_dir)
    metrics = {"cross_encoder": ce_name, "cross_encoder_is_finetuned": trained, "faithfulness": faith,
               "justification": just, "teacher_study": "not run (no teachers available); form written"}
    save_json(metrics, out / "metrics_xai.json")
    pd.DataFrame(rows).to_csv(out / "justification_samples.csv", index=False)
    pd.DataFrame([{"item_id": r["qid"], "student_answer": r["answer"], "explanation": r["text"] or r["template"],
                   "usefulness_1to5": "", "correct_yes_no": "", "comments": ""} for r in rows]
                 ).to_csv(out / "teacher_review_form.csv", index=False)
    log_experiment(cfg, "xai", metrics, config_sections=["xai"])


CFG = None

if __name__ == "__main__":
    main()

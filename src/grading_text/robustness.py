"""Adversarial / robustness probes for text grading built from real ASAG items.

  keyword_list : the reference's key concepts as a bare list (no meaning)       -> should score LOW
  word_salad   : the reference's words shuffled                                  -> should score LOW
  misspelled   : a full-mark student answer with ~15% words misspelled           -> should stay HIGH
  correct      : full-mark student answers (natural paraphrases of the reference) -> HIGH
"""
from __future__ import annotations

import random

import pandas as pd


def misspell(text: str, rng: random.Random, rate: float = 0.15) -> str:
    out = []
    for w in text.split():
        if len(w) > 3 and rng.random() < rate:
            i = rng.randrange(1, len(w) - 1)
            op = rng.choice(["swap", "drop", "dup"])
            if op == "swap":
                w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
            elif op == "drop":
                w = w[:i] + w[i + 1:]
            else:
                w = w[:i] + w[i] + w[i:]
        out.append(w)
    return " ".join(out)


def build_probes(df: pd.DataFrame, concepts_fn, seed: int, n_items: int = 40) -> pd.DataFrame:
    rng = random.Random(seed)
    full = df[df.score >= 0.999]
    items = df.drop_duplicates("qid").sample(n=min(n_items, df.qid.nunique()), random_state=seed)
    rows = []
    for _, it in items.iterrows():
        base = {"qid": it.qid, "question": it.question, "reference": it.reference}
        cons = concepts_fn(it.reference)
        rows.append({**base, "probe": "keyword_list", "answer": ", ".join(cons)})
        words = it.reference.split()
        rng.shuffle(words)
        rows.append({**base, "probe": "word_salad", "answer": " ".join(words)})
        good = full[full.qid == it.qid]
        if len(good):
            g = good.sample(n=1, random_state=seed).iloc[0]
            rows.append({**base, "probe": "correct", "answer": g.answer})
            rows.append({**base, "probe": "misspelled", "answer": misspell(g.answer, rng)})
    return pd.DataFrame(rows)

"""Deletion faithfulness: deleting the top-attributed (helping) words should lower the score
more than deleting the same number of random words."""
from __future__ import annotations

import random

import numpy as np


def delete_words(answer: str, targets: list[str], k: int) -> str:
    """Remove the first occurrence of each of the first k target words (case-insensitive)."""
    toks = answer.split()
    for t in targets[:k]:
        for i, w in enumerate(toks):
            if w.strip(".,;:!?()\"'").lower() == t.strip(".,;:!?()\"'").lower():
                toks.pop(i)
                break
    return " ".join(toks)


def deletion_test(score_fn, first: str, answer: str, words: list[dict], ks: list[int],
                  rng: random.Random) -> dict:
    """score_fn(first, answer) -> float. Returns score drops for top-k vs random-k deletion."""
    base = score_fn(first, answer)
    ranked = [w["word"] for w in sorted(words, key=lambda w: -w["attr"]) if w["attr"] > 0]
    all_words = answer.split()
    out = {"base": base}
    for k in ks:
        if len(all_words) <= k:
            continue
        top = score_fn(first, delete_words(answer, ranked, k)) if ranked else base
        rand_targets = rng.sample(all_words, k)
        rnd = score_fn(first, delete_words(answer, rand_targets, k))
        out[f"drop_top{k}"] = base - top
        out[f"drop_rand{k}"] = base - rnd
    return out


def summarize(results: list[dict], ks: list[int]) -> dict:
    s = {"n": len(results)}
    for k in ks:
        t = [r[f"drop_top{k}"] for r in results if f"drop_top{k}" in r]
        r_ = [r[f"drop_rand{k}"] for r in results if f"drop_rand{k}" in r]
        if t:
            s[f"mean_drop_top{k}"] = float(np.mean(t))
            s[f"mean_drop_random{k}"] = float(np.mean(r_))
            s[f"top_beats_random_frac{k}"] = float(np.mean([a > b for a, b in zip(t, r_)]))
    return s

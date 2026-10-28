"""CER / WER with consistent normalisation."""
from __future__ import annotations

import re

import jiwer


def normalize(t: str) -> str:
    t = t.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", t).strip()


def cer_wer(refs: list[str], hyps: list[str]) -> dict[str, float]:
    refs = [normalize(r) for r in refs]
    hyps = [normalize(h) for h in hyps]
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    if not pairs:
        return {"cer": float("nan"), "wer": float("nan"), "n": 0}
    r, h = zip(*pairs)
    return {"cer": jiwer.cer(list(r), list(h)), "wer": jiwer.wer(list(r), list(h)), "n": len(pairs)}

"""IAM handwriting lines (Teklia/IAM-line on the HF Hub; official writer-disjoint splits)."""
from __future__ import annotations

from PIL import Image


def load_iam_lines(split: str = "test", limit: int | None = None, seed: int = 42) -> list[tuple[Image.Image, str]]:
    from datasets import load_dataset

    ds = load_dataset("Teklia/IAM-line", split=split)
    if limit is not None and limit < len(ds):
        ds = ds.shuffle(seed=seed).select(range(limit))
    return [(r["image"].convert("RGB"), r["text"]) for r in ds]

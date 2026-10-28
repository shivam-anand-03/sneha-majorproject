"""Public short-answer grading datasets in one schema.

Unified columns: dataset, qid, question, reference, answer, score (normalised 0..1),
max_marks, group (student id used for leakage-free splits), split.

ASAP-SAS is not used for the cross-encoder: it ships rubrics but no reference
answers, so it cannot form (reference, answer) pairs.
"""
from __future__ import annotations

import re

import pandas as pd

from src.common.config import Config
from src.data.splits import assert_no_group_leakage, group_split

_STOP = re.compile(r"\s*<STOP>\s*")


def _clean(t: str) -> str:
    return _STOP.sub(" ", str(t)).strip()


def load_mohler(cfg: Config) -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset("nkazi/MohlerASAG", "cleaned", split="open_ended")
    df = ds.to_pandas()
    # id like E01.Q01.A00 -> A00 is the student; split by student
    parts = df["id"].str.split(".", expand=True)
    out = pd.DataFrame({
        "dataset": "mohler",
        "qid": parts[0] + "." + parts[1],
        "question": df["question"].map(_clean),
        "reference": df["instructor_answer"].map(_clean),
        "answer": df["student_answer"].map(_clean),
        "score": df["score_avg"].astype(float) / 5.0,
        "max_marks": 5.0,
        "group": "mohler_" + parts[2],
    })
    splits = group_split(out, "group", test_frac=0.2, seed=cfg.project.seed, val_frac=cfg.grading_text.cross_encoder.val_frac)
    assert_no_group_leakage(splits, "group")
    return pd.concat([d.assign(split=k) for k, d in splits.items()], ignore_index=True)


def _load_semeval(cfg: Config, name: str) -> pd.DataFrame:
    from datasets import load_dataset

    lmap = cfg.grading_text.cross_encoder.scientsbank_label_map
    ds = load_dataset(f"nkazi/{name}")
    frames = []
    for split_name in ["train", "test_ua", "test_uq"]:
        if split_name not in ds:
            continue
        d = ds[split_name]
        names = d.features["label"].names
        df = d.to_pandas()
        lab = df["label"].map(lambda i: names[i])
        frames.append(pd.DataFrame({
            "dataset": name.lower(),
            "qid": df["id"].str.rsplit(".", n=1).str[0] if name == "SciEntsBank" else df["id"].str.split(".").str[:2].str.join("."),
            "question": df["question"],
            "reference": df["reference_answer"],
            "answer": df["student_answer"],
            "score": lab.map(lambda l: float(lmap.get(l, 0.0))),
            "max_marks": 1.0,
            "group": name.lower() + "_" + df["id"],
            "split": {"train": "train", "test_ua": "test", "test_uq": "test_uq"}[split_name],
            "label": lab,
        }))
    df = pd.concat(frames, ignore_index=True)
    # carve a validation set from train, grouped by question so val != train answers
    tr = df[df.split == "train"]
    val_q = tr["qid"].drop_duplicates().sample(frac=cfg.grading_text.cross_encoder.val_frac,
                                               random_state=cfg.project.seed)
    df.loc[(df.split == "train") & df.qid.isin(val_q), "split"] = "val"
    return df


def load_asag(cfg: Config, names: list[str] | None = None) -> pd.DataFrame:
    names = names or list(cfg.grading_text.cross_encoder.datasets)
    loaders = {
        "mohler": load_mohler,
        "scientsbank": lambda c: _load_semeval(c, "SciEntsBank"),
        "beetle": lambda c: _load_semeval(c, "Beetle"),
    }
    df = pd.concat([loaders[n](cfg) for n in names], ignore_index=True)
    return df.dropna(subset=["reference", "answer"]).reset_index(drop=True)

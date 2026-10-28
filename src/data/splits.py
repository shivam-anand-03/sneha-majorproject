"""Group-level splits (by student / answer sheet) so no writer appears in two splits."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def group_split(df: pd.DataFrame, group_col: str, test_frac: float, seed: int,
                val_frac: float = 0.0) -> dict[str, pd.DataFrame]:
    """Split rows so every value of group_col lands in exactly one split."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    tr_idx, te_idx = next(gss.split(df, groups=df[group_col]))
    out = {"train": df.iloc[tr_idx], "test": df.iloc[te_idx]}
    if val_frac > 0:
        tr = out["train"]
        rel = val_frac / (1 - test_frac)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=rel, random_state=seed + 1)
        a, b = next(gss2.split(tr, groups=tr[group_col]))
        out["train"], out["val"] = tr.iloc[a], tr.iloc[b]
    return {k: v.reset_index(drop=True) for k, v in out.items()}


def assert_no_group_leakage(splits: dict[str, pd.DataFrame], group_col: str) -> None:
    seen: dict[str, str] = {}
    for name, d in splits.items():
        for g in np.unique(d[group_col]):
            if g in seen and seen[g] != name:
                raise AssertionError(f"group {g!r} in both {seen[g]} and {name}")
            seen[g] = name

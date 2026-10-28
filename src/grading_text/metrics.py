"""Agreement metrics vs teacher marks: Pearson, Spearman, QWK, MAE, RMSE, % within tolerance."""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score


def grading_metrics(pred_marks, true_marks, max_marks, step: float = 0.5, tol: float = 0.5) -> dict[str, float]:
    """pred/true in marks (not normalised). QWK computed on marks binned to `step`."""
    p = np.asarray(pred_marks, float)
    t = np.asarray(true_marks, float)
    mm = np.broadcast_to(np.asarray(max_marks, float), p.shape)
    pb = np.round(np.clip(p, 0, mm) / step).astype(int)
    tb = np.round(t / step).astype(int)
    out = {
        "n": int(len(p)),
        "pearson": float(pearsonr(p, t)[0]) if len(p) > 2 and p.std() > 0 and t.std() > 0 else float("nan"),
        "spearman": float(spearmanr(p, t)[0]) if len(p) > 2 and p.std() > 0 and t.std() > 0 else float("nan"),
        "qwk": float(cohen_kappa_score(tb, pb, weights="quadratic")) if len(set(tb) | set(pb)) > 1 else float("nan"),
        "mae": float(np.mean(np.abs(p - t))),
        "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
        f"within_{tol}": float(np.mean(np.abs(p - t) <= tol + 1e-9)),
    }
    return out

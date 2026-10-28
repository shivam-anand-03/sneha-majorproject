"""Line segmentation recall / precision against manually marked boxes (greedy IoU matching)."""
from __future__ import annotations

Box = tuple[int, int, int, int]


def iou(a: Box, b: Box) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match_boxes(pred: list[Box], gt: list[Box], thr: float) -> list[tuple[int, int, float]]:
    pairs = sorted(((iou(p, g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt)), reverse=True)
    used_p, used_g, out = set(), set(), []
    for s, i, j in pairs:
        if s < thr:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        out.append((i, j, s))
    return out


def line_pr(pred: list[Box], gt: list[Box], thr: float) -> dict[str, float]:
    m = match_boxes(pred, gt, thr)
    tp = len(m)
    recall = tp / len(gt) if gt else 1.0
    precision = tp / len(pred) if pred else (1.0 if not gt else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "n_pred": len(pred), "n_gt": len(gt), "recall": recall, "precision": precision, "f1": f1}

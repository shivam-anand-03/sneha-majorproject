"""Text / diagram separation with page-relative thresholds (replaces the paper's fixed w>40,h>400).

Idea: handwriting produces many small connected components no taller than a text line;
diagrams produce at least one tall, sparse component (box outlines, axes, arrows).
Tall sparse components seed diagram regions; nearby seeds are merged and every small
component inside the region (labels) is absorbed into the diagram.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.common.config import Config

Box = tuple[int, int, int, int]  # x0, y0, x1, y1


@dataclass
class Layout:
    text_mask: np.ndarray
    diagram_boxes: list[Box]


def _merge_boxes(boxes: list[Box], gap: int) -> list[Box]:
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        out: list[list[int]] = []
        for b in boxes:
            for o in out:
                if b[0] <= o[2] + gap and o[0] <= b[2] + gap and b[1] <= o[3] + gap and o[1] <= b[3] + gap:
                    o[0], o[1], o[2], o[3] = min(o[0], b[0]), min(o[1], b[1]), max(o[2], b[2]), max(o[3], b[3])
                    changed = True
                    break
            else:
                out.append(b)
        boxes = out
    return [tuple(b) for b in boxes]


def _hollow_components(binary: np.ndarray, labels: np.ndarray, stats: np.ndarray, L, page_area: int) -> set[int]:
    """Closed outlines (boxes, circles): enclosed area >> ink area. Letters like 'o' are hollow
    too but tiny, so a page-relative minimum enclosed area is required."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: set[int] = set()
    for c in contours:
        enclosed = cv2.contourArea(c)
        if enclosed < L.min_hollow_area_frac * page_area:
            continue
        x, y = c[0][0]
        lab = int(labels[y, x])
        if lab and enclosed / max(stats[lab, cv2.CC_STAT_AREA], 1) >= L.hollow_ratio:
            out.add(lab)
    return out


def separate(binary: np.ndarray, cfg: Config, enabled: bool = True) -> Layout:
    if not enabled:
        return Layout(binary.copy(), [])
    L = cfg.layout
    H, W = binary.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    max_text_h = L.max_text_line_height_frac * H
    hollow = _hollow_components(binary, labels, stats, L, H * W)
    seeds: list[Box] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        tall_sparse = (h > max_text_h and area / float(w * h) <= L.text_density_max
                       and w >= L.min_diagram_width_frac * W * 0.5)
        if tall_sparse or i in hollow:
            seeds.append((x, y, x + w, y + h))
    gap = int(L.merge_gap_frac * max(H, W))
    regions = _merge_boxes(seeds, gap)
    pad = int(L.diagram_pad_frac * max(H, W))
    diagrams: list[Box] = []
    for x0, y0, x1, y1 in regions:
        w, h = x1 - x0, y1 - y0
        if w * h < L.min_diagram_area_frac * H * W or h < L.min_diagram_height_frac * H or w < L.min_diagram_width_frac * W:
            continue
        diagrams.append((max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)))
    # absorb components (labels) whose centre lies inside a diagram box; also grow the box to contain them
    text_mask = binary.copy()
    final: list[Box] = []
    for x0, y0, x1, y1 in diagrams:
        bx0, by0, bx1, by1 = x0, y0, x1, y1
        for i in range(1, n):
            x, y, w, h, _ = stats[i]
            cx, cy = x + w / 2, y + h / 2
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                bx0, by0, bx1, by1 = min(bx0, x), min(by0, y), max(bx1, x + w), max(by1, y + h)
        final.append((bx0, by0, bx1, by1))
        text_mask[by0:by1, bx0:bx1] = 0
    return Layout(text_mask, final)

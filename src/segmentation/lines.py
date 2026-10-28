"""Text line segmentation: horizontal projection profile (baseline) or PaddleOCR DBNet detection."""
from __future__ import annotations

import numpy as np

from src.common.config import Config

Box = tuple[int, int, int, int]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as (start, end_exclusive)."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _split_tall(prof: np.ndarray, s: int, e: int, max_h: int, min_h: int) -> list[tuple[int, int]]:
    """Recursively split a band taller than max_h at its weakest interior row."""
    if e - s <= max_h:
        return [(s, e)]
    lo, hi = s + min_h, e - min_h
    if hi <= lo:
        return [(s, e)]
    cut = lo + int(np.argmin(prof[lo:hi]))
    return _split_tall(prof, s, cut, max_h, min_h) + _split_tall(prof, cut, e, max_h, min_h)


def _typical_line_height(text_mask: np.ndarray, bands: list, min_h: int, p, H: int) -> float:
    """Line height estimate for splitting touching lines. Band heights alone fail when most bands
    are merged pairs, so the upper quartile of word/letter component heights is used as well."""
    import cv2

    heights = [e - s for s, e in bands if e - s >= min_h]
    band_med = float(np.median(heights)) if heights else p.max_line_height_frac * H
    _, _, stats, _ = cv2.connectedComponentsWithStats(text_mask, 8)
    ch = stats[1:, cv2.CC_STAT_HEIGHT]
    ch = ch[ch >= max(min_h // 2, 2)]
    if len(ch) < 5:
        return band_med
    return min(band_med, p.cc_line_factor * float(np.percentile(ch, 75)))


def segment_projection(text_mask: np.ndarray, cfg: Config) -> list[Box]:
    p = cfg.segmentation.projection
    H, W = text_mask.shape
    prof = (text_mask > 0).sum(axis=1).astype(np.float64)
    if prof.max() == 0:
        return []
    k = max(1, int(p.smooth_window))
    prof_s = np.convolve(prof, np.ones(k) / k, mode="same")
    ink = prof_s > p.ink_threshold_frac * prof_s.max()
    # close tiny gaps (descenders / dots)
    bands = _runs(ink)
    merged: list[list[int]] = []
    for s, e in bands:
        if merged and s - merged[-1][1] < p.min_gap_rows:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    min_h = int(p.min_line_height_frac * H)
    typical = _typical_line_height(text_mask, merged, min_h, p, H)
    max_h = int(min(p.max_line_height_frac * H, p.split_factor * typical))
    pad = int(p.pad_px_frac * H)
    boxes: list[Box] = []
    for s, e in merged:
        for s2, e2 in _split_tall(prof_s, s, e, max_h, max(min_h, 1)):
            if e2 - s2 < min_h:
                continue
            band = text_mask[s2:e2]
            cols = (band > 0).sum(axis=0)
            xs = np.where(cols > p.min_col_ink_frac * (e2 - s2))[0]
            if len(xs) == 0:
                continue
            y0, y1 = max(0, s2 - pad), min(H, e2 + pad)
            boxes.append((max(0, int(xs[0]) - pad), int(y0), min(W, int(xs[-1]) + pad), int(y1)))
    return boxes


def segment_paddle(gray: np.ndarray, text_mask: np.ndarray, cfg: Config) -> list[Box]:
    """DBNet text detection via PaddleOCR, word boxes merged into lines by vertical overlap."""
    from paddleocr import TextDetection

    global _PADDLE_DET
    if "_PADDLE_DET" not in globals():
        _PADDLE_DET = TextDetection(model_name=cfg.segmentation.paddle.det_model, enable_mkldnn=False)
    img = np.stack([gray] * 3, -1)
    res = _PADDLE_DET.predict(img)[0]
    polys = res["dt_polys"]
    words: list[Box] = []
    for poly in polys:
        poly = np.asarray(poly)
        x0, y0 = poly.min(0)
        x1, y1 = poly.max(0)
        # drop detections that fall inside diagram areas (zeroed in text_mask)
        if text_mask[int(y0):int(y1), int(x0):int(x1)].sum() == 0:
            continue
        words.append((int(x0), int(y0), int(x1), int(y1)))
    words.sort(key=lambda b: (b[1] + b[3]) / 2)
    lines: list[list[int]] = []
    thr = cfg.segmentation.paddle.merge_y_overlap
    for b in words:
        for ln in lines:
            ov = min(ln[3], b[3]) - max(ln[1], b[1])
            if ov > thr * min(ln[3] - ln[1], b[3] - b[1]):
                ln[0], ln[1], ln[2], ln[3] = min(ln[0], b[0]), min(ln[1], b[1]), max(ln[2], b[2]), max(ln[3], b[3])
                break
        else:
            lines.append(list(b))
    return [tuple(l) for l in sorted(lines, key=lambda l: l[1])]


def segment_lines(gray: np.ndarray, text_mask: np.ndarray, cfg: Config, method: str | None = None) -> list[Box]:
    method = method or cfg.segmentation.method
    if method == "paddle":
        return segment_paddle(gray, text_mask, cfg)
    return segment_projection(text_mask, cfg)


def crop(img: np.ndarray, box: Box) -> np.ndarray:
    x0, y0, x1, y1 = box
    return img[y0:y1, x0:x1]

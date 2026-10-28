"""Page preprocessing.

Produces two images:
  * ``clean``  - grayscale, shadow-flattened, denoised, deskewed, ruled lines removed,
                 contrast-enhanced. This is what OCR sees (never binarised: TrOCR and
                 VLMs are trained on natural grayscale/RGB ink, binarisation hurts them).
  * ``binary`` - adaptive-threshold ink mask (ink=255). Used only for segmentation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.common.config import Config


@dataclass
class Preprocessed:
    clean: np.ndarray
    binary: np.ndarray
    skew_angle: float
    ruled_lines_removed: int
    steps: list[str] = field(default_factory=list)


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img.copy()


def remove_shadow(gray: np.ndarray, cfg: Config) -> np.ndarray:
    s = cfg.preprocess.shadow_removal
    bg = cv2.dilate(gray, np.ones((s.dilate_kernel, s.dilate_kernel), np.uint8))
    bg = cv2.medianBlur(bg, s.median_blur)
    diff = 255 - cv2.absdiff(gray, bg)
    return cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)


def denoise(gray: np.ndarray, cfg: Config) -> np.ndarray:
    d = cfg.preprocess.denoise
    return cv2.fastNlMeansDenoising(gray, None, d.h, d.template_window, d.search_window)


def clahe(gray: np.ndarray, cfg: Config) -> np.ndarray:
    c = cfg.preprocess.clahe
    return cv2.createCLAHE(clipLimit=c.clip_limit, tileGridSize=(c.tile_grid, c.tile_grid)).apply(gray)


def binarize(gray: np.ndarray, cfg: Config) -> np.ndarray:
    b = cfg.preprocess.binarize
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                 b.block_size, b.C)


def rotate(img: np.ndarray, angle: float, border: int = 255) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=border)


def _profile_score(bin_small: np.ndarray, angle: float) -> float:
    rot = rotate(bin_small, angle, border=0)
    prof = rot.sum(axis=1, dtype=np.float64)
    return float(np.sum(np.diff(prof) ** 2))


def estimate_skew_projection(binary: np.ndarray, cfg: Config) -> float:
    """Angle maximising sharpness of the horizontal projection profile (coarse then fine)."""
    d = cfg.preprocess.deskew
    scale = 1000 / max(binary.shape)
    small = cv2.resize(binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else binary
    coarse = np.arange(-d.max_angle, d.max_angle + 1e-6, d.coarse_step)
    best = max(coarse, key=lambda a: _profile_score(small, a))
    fine = np.arange(best - d.coarse_step, best + d.coarse_step + 1e-6, d.fine_step)
    return float(max(fine, key=lambda a: _profile_score(small, a)))


def estimate_skew_hough(binary: np.ndarray, cfg: Config) -> float:
    d = cfg.preprocess.deskew
    edges = cv2.Canny(binary, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 1800, d.hough_threshold)
    if lines is None:
        return 0.0
    angles = []
    for rho_theta in lines[:, 0]:
        deg = np.degrees(rho_theta[1]) - 90
        if abs(deg) <= d.max_angle:
            angles.append(deg)
    return float(np.median(angles)) if angles else 0.0


def remove_ruled_lines(gray: np.ndarray, binary: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray, int]:
    """Detect long horizontal rules by morphological opening; inpaint them out of gray and mask."""
    r = cfg.preprocess.ruled_lines
    w = gray.shape[1]
    k = max(15, int(w * r.kernel_frac))
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(horiz, 8)
    mask = np.zeros_like(binary)
    count = 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_WIDTH] >= r.min_line_frac * w:
            mask[labels == i] = 255
            count += 1
    if count == 0:
        return gray, binary, 0
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    gray2 = cv2.inpaint(gray, mask, r.inpaint_radius, cv2.INPAINT_TELEA)
    bin2 = cv2.bitwise_and(binary, cv2.bitwise_not(mask))
    # reconnect strokes that crossed the rule
    bin2 = cv2.morphologyEx(bin2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)))
    return gray2, bin2, count


def preprocess_page(img: np.ndarray, cfg: Config, enabled: bool | None = None) -> Preprocessed:
    p = cfg.preprocess
    gray = to_gray(img)
    if not (p.enabled if enabled is None else enabled):
        return Preprocessed(gray, binarize(gray, cfg), 0.0, 0, ["grayscale"])
    steps = ["grayscale"]
    if p.shadow_removal.enabled:
        gray = remove_shadow(gray, cfg)
        steps.append("shadow_removal")
    if p.denoise.enabled:
        gray = denoise(gray, cfg)
        steps.append("denoise")
    angle = 0.0
    if p.deskew.enabled:
        b = binarize(gray, cfg)
        angle = estimate_skew_projection(b, cfg) if p.deskew.method == "projection" else estimate_skew_hough(b, cfg)
        if abs(angle) > 0.05:
            gray = rotate(gray, angle)
        steps.append(f"deskew({angle:+.2f})")
    binary = binarize(gray, cfg)
    n_rules = 0
    if p.ruled_lines.enabled:
        gray, binary, n_rules = remove_ruled_lines(gray, binary, cfg)
        steps.append(f"ruled_lines({n_rules})")
    if p.clahe.enabled:
        gray = clahe(gray, cfg)
        steps.append("clahe")
    # despeckle the segmentation mask
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return Preprocessed(gray, binary, angle, n_rules, steps)

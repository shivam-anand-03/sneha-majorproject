import random

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from src.data.synthetic import font_paths
from src.grading_diagram import synth
from src.preprocess.image import preprocess_page, rotate
from src.segmentation.layout import separate
from src.segmentation.lines import segment_projection
from src.segmentation.metrics import iou, line_pr


def _page(cfg, n_lines=4, ruled=True, diagram=False):
    W, H = 1240, 1754  # A4 @150dpi keeps tests fast
    img = Image.new("RGB", (W, H), (245, 245, 240))
    d = ImageDraw.Draw(img)
    sp = 70
    if ruled:
        for y in range(120, H - 60, sp):
            d.line([(0, y), (W, y)], fill=(170, 190, 225), width=2)
    font = ImageFont.truetype(font_paths(cfg)[0], 40)
    gt = []
    for i in range(n_lines):
        y = 120 + sp * (i + 1) - 8
        d.text((100, y), "the quick brown fox jumps over", font=font, fill=(20, 20, 80), anchor="ls")
        gt.append(d.textbbox((100, y), "the quick brown fox jumps over", font=font, anchor="ls"))
    if diagram:
        spec = synth.reference_spec(random.Random(0))
        dimg = synth.render(spec, font_paths(cfg)[0], (800, 380), random.Random(1))
        img.paste(Image.fromarray(dimg), (220, 1000))
        ys, xs = np.where(dimg.min(-1) < 160)
        gt.append((220 + xs.min(), 1000 + ys.min(), 220 + xs.max(), 1000 + ys.max()))
    return np.array(img), gt


def test_deskew_recovers_angle(cfg):
    img, _ = _page(cfg)
    skewed = rotate(img, -4.0)
    pre = preprocess_page(skewed, cfg)
    assert abs(pre.skew_angle - 4.0) < 0.3


def test_ruled_lines_removed_and_text_kept(cfg):
    img, gt = _page(cfg, ruled=True)
    pre = preprocess_page(img, cfg)
    assert pre.ruled_lines_removed >= 10
    lay = separate(pre.binary, cfg)
    boxes = segment_projection(lay.text_mask, cfg)
    assert line_pr(boxes, [tuple(b) for b in gt], 0.5)["recall"] == 1.0


def test_ocr_image_is_not_binarised(cfg):
    img, _ = _page(cfg)
    pre = preprocess_page(img, cfg)
    assert len(np.unique(pre.clean)) > 20  # grayscale retained for OCR


def test_diagram_detected_with_relative_thresholds(cfg):
    img, gt = _page(cfg, diagram=True)
    pre = preprocess_page(img, cfg)
    lay = separate(pre.binary, cfg)
    assert len(lay.diagram_boxes) == 1
    assert iou(lay.diagram_boxes[0], gt[-1]) > 0.6
    # diagram labels must not appear as text lines
    boxes = segment_projection(lay.text_mask, cfg)
    assert all(b[3] < gt[-1][1] or b[1] > gt[-1][3] for b in boxes)


def test_diagram_detection_scale_invariant(cfg):
    img, _ = _page(cfg, diagram=True)
    small = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    pre = preprocess_page(small, cfg)
    assert len(separate(pre.binary, cfg).diagram_boxes) == 1


def test_line_pr_metric():
    m = line_pr([(0, 0, 10, 10), (50, 50, 60, 60)], [(0, 0, 10, 10)], 0.5)
    assert m["recall"] == 1.0 and m["precision"] == 0.5

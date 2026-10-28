"""Page -> preprocessed image, diagram regions, text line boxes and crops."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.common.config import Config
from src.preprocess.image import Preprocessed, preprocess_page
from src.segmentation.layout import separate
from src.segmentation.lines import crop, segment_lines

Box = tuple[int, int, int, int]


@dataclass
class PageAnalysis:
    pre: Preprocessed
    line_boxes: list[Box]
    diagram_boxes: list[Box]
    line_crops: list[np.ndarray]
    diagram_crops: list[np.ndarray]


def analyze_page(img: np.ndarray, cfg: Config, preprocess: bool | None = None, split_diagrams: bool = True,
                 seg_method: str | None = None) -> PageAnalysis:
    pre = preprocess_page(img, cfg, enabled=preprocess)
    lay = separate(pre.binary, cfg, enabled=split_diagrams)
    boxes = segment_lines(pre.clean, lay.text_mask, cfg, method=seg_method)
    return PageAnalysis(
        pre=pre, line_boxes=boxes, diagram_boxes=lay.diagram_boxes,
        line_crops=[crop(pre.clean, b) for b in boxes],
        diagram_crops=[crop(pre.clean, b) for b in lay.diagram_boxes],
    )


def save_debug(pa: PageAnalysis, out_dir: str | Path, stem: str) -> Path:
    """Write the cleaned page with boxes drawn, plus every line / diagram crop."""
    out = Path(out_dir) / stem
    out.mkdir(parents=True, exist_ok=True)
    vis = cv2.cvtColor(pa.pre.clean, cv2.COLOR_GRAY2BGR)
    for i, (x0, y0, x1, y1) in enumerate(pa.line_boxes):
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 160, 0), 3)
        cv2.putText(vis, str(i), (x0, max(30, y0)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 160, 0), 2)
        cv2.imwrite(str(out / f"line_{i:02d}.png"), pa.line_crops[i])
    for i, (x0, y0, x1, y1) in enumerate(pa.diagram_boxes):
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 220), 4)
        cv2.imwrite(str(out / f"diagram_{i}.png"), pa.diagram_crops[i])
    cv2.imwrite(str(out / "page_boxes.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
    cv2.imwrite(str(out / "mask.png"), pa.pre.binary)
    return out

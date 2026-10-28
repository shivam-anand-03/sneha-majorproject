"""PDF / image input -> list of RGB numpy pages at the configured DPI."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.common.config import Config

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def pdf_to_images(path: str | Path, cfg: Config) -> list[np.ndarray]:
    path = Path(path)
    dpi = cfg.pdf.dpi
    if cfg.pdf.backend == "pdf2image":
        from pdf2image import convert_from_path

        return [np.array(p.convert("RGB")) for p in convert_from_path(str(path), dpi=dpi)]
    import fitz

    pages = []
    with fitz.open(path) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
            pages.append(np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, 3).copy())
    return pages


def load_pages(path: str | Path, cfg: Config) -> list[np.ndarray]:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return pdf_to_images(path, cfg)
    if path.suffix.lower() in IMAGE_EXT:
        return [np.array(Image.open(path).convert("RGB"))]
    raise ValueError(f"unsupported input type {path.suffix}")


def images_to_pdf(images: list[np.ndarray], out: str | Path, dpi: int) -> None:
    pil = [Image.fromarray(im).convert("RGB") for im in images]
    pil[0].save(out, save_all=True, append_images=pil[1:], resolution=dpi)

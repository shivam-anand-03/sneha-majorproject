"""Image-embedding similarity for diagrams (CLIP / SigLIP / DINOv2) + the paper's sigmoid transform.

The paper's T(x) = 0.5 + 0.5/(1+exp(-k(x-c))) maps any similarity to [0.5, 1], so even an
unrelated drawing gets at least half marks. We expose both T and the rescaled
T01(x) = 2*(T(x)-0.5) in [0, 1]; the scorer uses T01 (see scripts/eval_diagram.py for the
measured comparison).
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from src.common.config import Config
from src.common.gpu import free_memory, get_device, is_oom

_MODELS: dict[str, tuple] = {}


def paper_transform(x, k: float, c: float):
    return 0.5 + 0.5 / (1.0 + np.exp(-k * (np.asarray(x, float) - c)))


def transform01(x, k: float, c: float):
    return 2.0 * (paper_transform(x, k, c) - 0.5)


def prepare(img: np.ndarray, size: int) -> Image.Image:
    """Pad to square on white (keeps aspect, so structure isn't distorted) and resize."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    h, w = img.shape[:2]
    s = max(h, w)
    canvas = np.full((s, s, 3), 255, np.uint8)
    canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = img[..., :3]
    return Image.fromarray(canvas).resize((size, size), Image.BICUBIC)


def _model_name(kind: str, cfg: Config) -> str:
    g = cfg.grading_diagram
    return {"clip": g.clip_model, "siglip": g.siglip_model, "dino": g.dino_model}[kind]


def get_model(kind: str, cfg: Config):
    if kind not in _MODELS:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        device = get_device(cfg.project.device)
        dtype = torch.float16 if device == "cuda" and cfg.project.fp16 else torch.float32
        name = _model_name(kind, cfg)
        model = AutoModel.from_pretrained(name, torch_dtype=dtype).to(device).eval()
        proc = AutoImageProcessor.from_pretrained(name)
        _MODELS[kind] = (model, proc, device, dtype)
    return _MODELS[kind]


def _embed_once(kind: str, images: list[Image.Image], cfg: Config) -> np.ndarray:
    import torch

    model, proc, device, dtype = get_model(kind, cfg)
    px = proc(images=images, return_tensors="pt")["pixel_values"].to(device, dtype)
    with torch.inference_mode():
        if kind in ("clip", "siglip"):
            feats = model.get_image_features(pixel_values=px)
        else:  # DINOv2: CLS token
            feats = model(pixel_values=px).last_hidden_state[:, 0]
    feats = torch.nn.functional.normalize(feats.float(), dim=-1)
    return feats.cpu().numpy()


def embed(kind: str, imgs: list[np.ndarray], cfg: Config) -> np.ndarray:
    pil = [prepare(i, cfg.grading_diagram.render_size) for i in imgs]
    out, bs = [], cfg.gpu.oom_retry_batch_sizes[0]
    i = 0
    while i < len(pil):
        try:
            out.append(_embed_once(kind, pil[i:i + bs], cfg))
            i += bs
        except RuntimeError as e:
            if not is_oom(e) or bs == 1:
                raise
            free_memory()
            bs = max(1, bs // 2)
    return np.concatenate(out, 0)


def similarities(student: np.ndarray, reference: np.ndarray, cfg: Config,
                 kinds: list[str] | None = None) -> dict[str, float]:
    kinds = kinds or list(cfg.grading_diagram.image_embedders)
    out = {}
    for k in kinds:
        e = embed(k, [student, reference], cfg)
        out[k] = float(np.clip(e[0] @ e[1], 0.0, 1.0))
    return out


def blend(sims: dict[str, float], weights: dict[str, float]) -> float:
    ws = {k: float(weights.get(k, 0.0)) for k in sims}
    tot = sum(ws.values())
    if tot <= 0:  # unweighted mean if weights don't cover the chosen embedders
        return float(np.mean(list(sims.values())))
    return float(sum(sims[k] * w for k, w in ws.items()) / tot)

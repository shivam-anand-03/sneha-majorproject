"""Grad-CAM-style heatmap for diagram similarity.

Target = cosine(student embedding, reference embedding). Activations are the patch tokens
entering the last transformer block (the final CLS depends on them through attention);
cam = ReLU(sum_d grad * activation), reshaped to the patch grid, upsampled and overlaid on
the student crop. Bright regions are the parts of the drawing that drove the similarity.
The map is restricted to (dilated) ink so background artefact tokens cannot dominate it.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.common.config import Config
from src.common.gpu import get_device

_MODELS: dict[str, tuple] = {}


def _load(cfg: Config, which: str):
    if which not in _MODELS:
        from transformers import AutoImageProcessor, AutoModel, CLIPVisionModelWithProjection

        name = cfg.grading_diagram.clip_model if which == "clip" else cfg.grading_diagram.dino_model
        proc = AutoImageProcessor.from_pretrained(name)
        model = (CLIPVisionModelWithProjection if which == "clip" else AutoModel).from_pretrained(name)
        model = model.to(get_device(cfg.project.device)).eval()
        _MODELS[which] = (model, proc)
    return _MODELS[which]


def _pixels(img: np.ndarray, proc, size: int, device):
    import torch

    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else img
    x = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = (x - np.array(proc.image_mean, np.float32)) / np.array(proc.image_std, np.float32)
    return torch.from_numpy(x.transpose(2, 0, 1))[None].to(device)


def _embed(model, which: str, pv):
    out = model(pixel_values=pv, output_hidden_states=True)
    emb = out.image_embeds if which == "clip" else out.pooler_output
    return emb, out.hidden_states[-2]


def heatmap(student: np.ndarray, reference: np.ndarray, cfg: Config, which: str | None = None,
            size: int = 224) -> dict:
    import torch

    which = which or cfg.xai.gradcam_target
    model, proc = _load(cfg, which)
    dev = next(model.parameters()).device
    with torch.no_grad():
        ref_emb, _ = _embed(model, which, _pixels(reference, proc, size, dev))
    emb, acts = _embed(model, which, _pixels(student, proc, size, dev))
    acts.retain_grad()
    sim = torch.nn.functional.cosine_similarity(emb, ref_emb).sum()
    model.zero_grad()
    sim.backward()
    patches_a, patches_g = acts[0, 1:], acts.grad[0, 1:]
    n = int(round(np.sqrt(patches_a.shape[0])))
    patches_a, patches_g = patches_a[-n * n:], patches_g[-n * n:]  # drop register tokens if any
    cam = torch.relu((patches_g * patches_a).sum(-1)).reshape(n, n).detach().float().cpu().numpy()
    h, w = student.shape[:2]
    cam_up = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, None) * ink_mask(student)
    hi = np.percentile(cam_up[cam_up > 0], 99) if (cam_up > 0).any() else 0.0
    cam_up = (cam_up / hi).clip(0, 1) if hi > 0 else cam_up
    return {"similarity": float(sim.item()), "heatmap": cam_up, "overlay": overlay(student, cam_up),
            "model": which}


def ink_mask(img: np.ndarray, dilate_frac: float = 0.03) -> np.ndarray:
    """Soft mask of drawn content. ViTs park global information in a few high-norm background
    tokens; without this mask those artefacts dominate the map although nothing is drawn there."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    ink = (gray < 160).astype(np.uint8)
    k = max(3, int(dilate_frac * max(gray.shape)))
    ink = cv2.dilate(ink, np.ones((k, k), np.uint8)).astype(np.float32)
    return cv2.GaussianBlur(ink, (0, 0), k / 2)


def overlay(img: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else img
    color = cv2.cvtColor(cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return (rgb * (1 - alpha) + color * alpha).astype(np.uint8)


def diagram_heatmap(student_crop: np.ndarray, ref_img: np.ndarray, cfg: Config) -> np.ndarray:
    """Pipeline entry point: RGB overlay of the similarity heatmap on the student diagram crop."""
    return heatmap(student_crop, ref_img, cfg)["overlay"]

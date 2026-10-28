"""Bi-encoder (SBERT) cosine similarity."""
from __future__ import annotations

import numpy as np

from src.common.config import Config
from src.common.gpu import get_device

_MODELS: dict[str, object] = {}


def get_embedder(name: str, cfg: Config):
    if name not in _MODELS:
        from sentence_transformers import SentenceTransformer

        m = SentenceTransformer(name, device=get_device(cfg.project.device))
        if cfg.project.fp16 and m.device.type == "cuda":
            m = m.half()
        _MODELS[name] = m
    return _MODELS[name]


def encode(texts: list[str], cfg: Config, name: str | None = None) -> np.ndarray:
    m = get_embedder(name or cfg.grading_text.primary_embedder, cfg)
    return m.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True,
                    show_progress_bar=False).astype(np.float32)


def pair_similarity(refs: list[str], answers: list[str], cfg: Config, name: str | None = None) -> np.ndarray:
    a = encode(refs, cfg, name)
    b = encode(answers, cfg, name)
    return np.clip((a * b).sum(1), 0.0, 1.0)

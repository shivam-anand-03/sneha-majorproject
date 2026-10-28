"""Cross-encoder short-answer grader: (question + reference, student answer) -> score in [0,1]."""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import Config, resolve
from src.common.gpu import get_device

log = logging.getLogger(__name__)
_CE: dict[str, object] = {}


def first_text(question: str, reference: str) -> str:
    q = (question or "").strip()
    return f"Question: {q} Reference answer: {reference}" if q else f"Reference answer: {reference}"


def model_path(cfg: Config) -> Path:
    return resolve(cfg, cfg.grading_text.cross_encoder.output_dir)


def train_cross_encoder(train: pd.DataFrame, val: pd.DataFrame, cfg: Config) -> Path:
    import torch
    from sentence_transformers import InputExample
    from sentence_transformers.cross_encoder import CrossEncoder
    from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
    from torch.utils.data import DataLoader

    c = cfg.grading_text.cross_encoder
    device = get_device(cfg.project.device)
    model = CrossEncoder(c.base_model, num_labels=1, max_length=c.max_length, device=device,
                         activation_fn=torch.nn.Sigmoid())
    ex = [InputExample(texts=[first_text(q, r), a], label=float(s))
          for q, r, a, s in zip(train.question, train.reference, train.answer, train.score)]
    dl = DataLoader(ex, shuffle=True, batch_size=c.batch_size)
    evaluator = CECorrelationEvaluator(
        [[first_text(q, r), a] for q, r, a in zip(val.question, val.reference, val.answer)],
        val.score.astype(float).tolist(), name="val")
    out = model_path(cfg)
    model.fit(train_dataloader=dl, evaluator=evaluator, epochs=c.epochs,
              warmup_steps=math.ceil(len(dl) * c.epochs * c.warmup_ratio),
              optimizer_params={"lr": c.lr}, output_path=str(out), save_best_model=True,
              use_amp=device == "cuda", show_progress_bar=True)
    if not (out / "config.json").exists():  # no evaluator improvement -> still save last
        model.save(str(out))
    return out


def get_cross_encoder(cfg: Config, path: str | Path | None = None):
    import torch
    from sentence_transformers.cross_encoder import CrossEncoder

    path = str(path or model_path(cfg))
    if path not in _CE:
        if not Path(path).exists():
            raise FileNotFoundError(f"cross-encoder not trained yet: {path} (run scripts/train_text_grader.py)")
        _CE[path] = CrossEncoder(path, device=get_device(cfg.project.device), activation_fn=torch.nn.Sigmoid(),
                                 max_length=cfg.grading_text.cross_encoder.max_length)
    return _CE[path]


def predict(questions: list[str], refs: list[str], answers: list[str], cfg: Config) -> np.ndarray:
    ce = get_cross_encoder(cfg)
    pairs = [[first_text(q, r), a] for q, r, a in zip(questions, refs, answers)]
    return np.asarray(ce.predict(pairs, batch_size=32, show_progress_bar=False), dtype=np.float32)

"""Config loading. All thresholds/weights/model names come from configs/config.yaml."""
from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


class Config(dict):
    """dict with attribute access (cfg.ocr.trocr.model) that stays YAML-serialisable."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __deepcopy__(self, memo):
        return Config(copy.deepcopy(dict(self), memo))


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _unwrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _unwrap(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unwrap(v) for v in obj]
    return obj


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    path = Path(path or os.environ.get("AES_CONFIG", DEFAULT_CONFIG))
    with open(path) as f:
        raw = yaml.safe_load(f)
    if overrides:
        _deep_update(raw, overrides)
    return _wrap(raw)


def to_plain(cfg: Config) -> dict:
    return _unwrap(cfg)


def resolve(cfg: Config, rel: str | Path) -> Path:
    """Resolve a config-relative path against the repo root."""
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

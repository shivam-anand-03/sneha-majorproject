"""Lightweight experiment tracking: one JSON line per run with config, model versions and metrics."""
from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from .config import Config, resolve, to_plain


def _git_rev() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _lib_versions() -> dict[str, str]:
    out = {}
    for name in ["torch", "transformers", "sentence_transformers", "cv2", "paddleocr", "lightgbm", "captum"]:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "?")
        except Exception:
            pass
    return out


def log_experiment(cfg: Config, name: str, metrics: dict[str, Any], extra: dict[str, Any] | None = None,
                   config_sections: list[str] | None = None) -> dict:
    """Append a run record. config_sections limits which config blocks are stored (default: all)."""
    plain = to_plain(cfg)
    if config_sections:
        plain = {k: plain[k] for k in config_sections if k in plain}
    rec = {
        "name": name,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git": _git_rev(),
        "python": platform.python_version(),
        "libs": _lib_versions(),
        "config": plain,
        "metrics": metrics,
        **(extra or {}),
    }
    path = resolve(cfg, cfg.paths.experiments_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return rec


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)

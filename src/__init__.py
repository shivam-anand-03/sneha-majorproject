"""Keep every download inside the project folder (models, datasets, OCR weights)."""
import os
from pathlib import Path

_CACHE = Path(__file__).resolve().parents[1] / ".cache"
for _var, _sub in {"HF_HOME": "huggingface", "PADDLE_PDX_CACHE_HOME": "paddlex", "TORCH_HOME": "torch",
                   "UV_CACHE_DIR": "uv", "MPLCONFIGDIR": "matplotlib"}.items():
    os.environ.setdefault(_var, str(_CACHE / _sub))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # xet stalled downloads on this machine

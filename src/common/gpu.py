"""Device selection and out-of-memory handling."""
from __future__ import annotations

import gc
import logging
from typing import Callable, Sequence, TypeVar

log = logging.getLogger(__name__)
T = TypeVar("T")


def get_device(pref: str = "auto") -> str:
    import torch

    if pref == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if pref == "cuda":
        log.warning("CUDA requested but unavailable; falling back to CPU")
    return "cpu"


def torch_dtype(device: str, fp16: bool):
    import torch

    if device != "cuda" or not fp16:
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def free_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def is_oom(err: BaseException) -> bool:
    msg = str(err).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg or type(err).__name__ == "OutOfMemoryError"


def run_batched_with_oom_fallback(
    fn: Callable[[Sequence], list[T]], items: Sequence, batch_sizes: Sequence[int]
) -> list[T]:
    """Run fn over items in batches; on CUDA OOM retry with the next smaller batch size."""
    for bs in batch_sizes:
        try:
            out: list[T] = []
            for i in range(0, len(items), bs):
                out.extend(fn(items[i : i + bs]))
            return out
        except RuntimeError as e:
            if not is_oom(e):
                raise
            log.warning("OOM at batch size %d, retrying smaller", bs)
            free_memory()
    raise RuntimeError(f"OOM even at batch size {batch_sizes[-1]}")

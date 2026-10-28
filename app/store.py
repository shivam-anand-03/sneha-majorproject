"""Result storage under outputs/app_results/{result_id}/ (result.json, inputs, crop images)."""
from __future__ import annotations

import json
import re
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path

from src.common.config import Config, resolve

_ID_RE = re.compile(r"^[a-f0-9]{12}$")
_FILE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")


class NotFound(Exception):
    pass


def results_root(cfg: Config) -> Path:
    root = resolve(cfg, cfg.project.output_dir) / "app_results"
    root.mkdir(parents=True, exist_ok=True)
    return root


def new_result_dir(cfg: Config) -> tuple[str, Path]:
    rid = uuid.uuid4().hex[:12]
    d = results_root(cfg) / rid
    d.mkdir(parents=True)
    return rid, d


def result_dir(cfg: Config, rid: str) -> Path:
    if not _ID_RE.match(rid):
        raise NotFound(rid)
    d = results_root(cfg) / rid
    if not (d / "result.json").exists():
        raise NotFound(rid)
    return d


def load_result(cfg: Config, rid: str) -> dict:
    return json.loads((result_dir(cfg, rid) / "result.json").read_text())


_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


@contextmanager
def result_lock(rid: str):
    """Serialise read-modify-write of one result (endpoints run concurrently in the threadpool)."""
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(rid, threading.Lock())
    with lock:
        yield


def save_result(d: Path, result: dict) -> None:
    tmp = d / f"result.json.{uuid.uuid4().hex[:8]}.tmp"
    tmp.write_text(json.dumps(result, indent=1, default=str, allow_nan=False))
    tmp.replace(d / "result.json")


def result_file(cfg: Config, rid: str, name: str) -> Path:
    """Only plain file names inside the result folder are served; never a client-built path."""
    if not _FILE_RE.match(name) or name.startswith(".") or name.endswith(".json"):
        raise NotFound(name)
    p = result_dir(cfg, rid) / name
    if not p.is_file():
        raise NotFound(name)
    return p

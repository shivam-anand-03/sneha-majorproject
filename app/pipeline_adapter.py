"""Chooses the real pipeline (src.pipeline.evaluate.evaluate_sheet) or the demo pipeline.

Both share the signature
    evaluate_sheet(sheet_path, answer_key, cfg, student_id_anon, topic, edited_lines, out_dir, prior)
and write every image they reference by name (page, crops, Grad-CAM) into out_dir.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable

from src.common.config import Config

log = logging.getLogger(__name__)


def _load() -> tuple[Callable, str]:
    if os.environ.get("AES_DEMO_PIPELINE") == "1":
        from app.mock_pipeline import evaluate_sheet

        return evaluate_sheet, "demo"
    try:
        from src.pipeline.evaluate import evaluate_sheet

        return evaluate_sheet, "full"
    except ImportError as e:
        log.warning("full pipeline unavailable (%s); using demo pipeline", e)
        from app.mock_pipeline import evaluate_sheet

        return evaluate_sheet, "demo"


_FN, PIPELINE_KIND = _load()


def run(sheet_path: Path | None, answer_key: dict, cfg: Config, out_dir: Path, student_id_anon: str | None,
        topic: str | None, edited_lines: list[dict] | None = None, prior: dict | None = None) -> dict:
    """Fresh evaluation (prior=None) or regrade of `prior` with teacher-edited lines (no re-OCR)."""
    result = _FN(sheet_path, answer_key, cfg, student_id_anon=student_id_anon, topic=topic,
                 edited_lines=edited_lines, out_dir=out_dir, prior=prior)
    result.setdefault("config_snapshot", {})["pipeline"] = PIPELINE_KIND
    return result

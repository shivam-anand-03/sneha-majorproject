"""Annotation CSVs for own data.

Lines CSV  : image_path, sheet_id, student_id_anonymized, line_idx, x0,y0,x1,y1,
             ocr_text, verified_transcription
Sheets CSV : sheet_id, student_id_anonymized, question_id, teacher_marks,
             teacher_diagram_marks, max_marks
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.config import Config
from src.data.anonymize import anonymize_frame
from src.data.splits import assert_no_group_leakage, group_split

LINE_COLS = ["image_path", "sheet_id", "student_id_anonymized", "line_idx", "x0", "y0", "x1", "y1",
             "ocr_text", "verified_transcription"]
SHEET_COLS = ["sheet_id", "student_id_anonymized", "question_id", "teacher_marks",
              "teacher_diagram_marks", "max_marks"]


def create_line_template(crops: list[dict], out_csv: str | Path) -> pd.DataFrame:
    """crops: dicts with image_path, sheet_id, student_id_anonymized, line_idx, bbox, ocr_text.
    verified_transcription is pre-filled with OCR text for the annotator to correct."""
    rows = []
    for c in crops:
        x0, y0, x1, y1 = c["bbox"]
        rows.append({**{k: c.get(k) for k in LINE_COLS}, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                     "verified_transcription": c.get("ocr_text", "")})
    df = pd.DataFrame(rows, columns=LINE_COLS)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def import_raw_sheet_marks(raw_csv: str | Path, cfg: Config, out_csv: str | Path) -> pd.DataFrame:
    """Import a teacher's marks sheet that contains real student ids/names; store only anonymised."""
    df = anonymize_frame(pd.read_csv(raw_csv), cfg)
    missing = [c for c in SHEET_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"marks csv missing columns {missing}")
    df[SHEET_COLS].to_csv(out_csv, index=False)
    return df[SHEET_COLS]


def load_verified_pairs(csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df = df.dropna(subset=["ocr_text", "verified_transcription"])
    return df[df["verified_transcription"].str.strip() != ""]


def split_lines_by_student(df: pd.DataFrame, cfg: Config, test_frac: float = 0.2) -> dict[str, pd.DataFrame]:
    splits = group_split(df, "student_id_anonymized", test_frac, cfg.project.seed, val_frac=0.1)
    assert_no_group_leakage(splits, "student_id_anonymized")
    return splits

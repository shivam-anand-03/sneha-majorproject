"""Anonymisation of student identifiers before anything is stored.

Student IDs are replaced by a salted SHA-256 hash (stable per salt, not reversible
without it). Free text is scrubbed of roll-number-like tokens and known names.
The salt is read from an environment variable and must never be committed.
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Iterable

import pandas as pd

from src.common.config import Config


def get_salt(cfg: Config) -> str:
    return os.environ.get(cfg.anonymization.salt_env_var, cfg.anonymization.default_salt)


def anon_id(raw_id: str, cfg: Config, salt: str | None = None) -> str:
    salt = salt if salt is not None else get_salt(cfg)
    h = hashlib.sha256(f"{salt}::{str(raw_id).strip().lower()}".encode()).hexdigest()
    return f"{cfg.anonymization.id_prefix}{h[: cfg.anonymization.hash_len]}"


def scrub_text(text: str, cfg: Config, names: Iterable[str] = ()) -> str:
    if not isinstance(text, str):
        return text
    for pat in cfg.anonymization.roll_number_patterns:
        text = re.sub(pat, "[ID]", text)
    for name in sorted({n for n in names if n and len(n) > 1}, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(name)}\b", "[NAME]", text, flags=re.IGNORECASE)
    return text


def anonymize_frame(df: pd.DataFrame, cfg: Config, id_col: str = "student_id",
                    name_cols: Iterable[str] = ("student_name", "name"),
                    text_cols: Iterable[str] = ("verified_transcription", "student_answer", "notes")) -> pd.DataFrame:
    """Return a copy with ids hashed, name columns dropped and free-text scrubbed."""
    df = df.copy()
    salt = get_salt(cfg)
    names: set[str] = set()
    for c in name_cols:
        if c in df.columns:
            for full in df[c].dropna().astype(str):
                names.add(full)
                names.update(full.split())
    if id_col in df.columns:
        df["student_id_anonymized"] = df[id_col].astype(str).map(lambda s: anon_id(s, cfg, salt))
        df = df.drop(columns=[id_col])
    df = df.drop(columns=[c for c in name_cols if c in df.columns])
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].map(lambda t: scrub_text(t, cfg, names))
    return df

"""Optional ByT5 OCR post-correction, gated on >= cfg.correction.min_pairs verified pairs.

Corrected text is never auto-trusted: callers get raw + corrected side by side and a
`meaning_changed` flag (SBERT similarity between raw and corrected below threshold).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.config import Config, resolve
from src.common.gpu import get_device


def corrector_dir(cfg: Config) -> Path:
    return resolve(cfg, cfg.project.models_dir) / "byt5_correction"


def correction_status(cfg: Config, pairs_csv: str | Path | None = None) -> dict:
    """Decide whether correction may run. Returns {enabled, reason, n_pairs}."""
    mode = cfg.correction.enabled
    n = 0
    if pairs_csv and Path(pairs_csv).exists():
        from src.data.annotations import load_verified_pairs

        n = len(load_verified_pairs(pairs_csv))
    trained = (corrector_dir(cfg) / "config.json").exists()
    if mode is False or mode == "off":
        return {"enabled": False, "reason": "disabled in config", "n_pairs": n}
    if not trained and n < cfg.correction.min_pairs:
        return {"enabled": False, "n_pairs": n,
                "reason": f"only {n} verified pairs (< {cfg.correction.min_pairs}); correction skipped"}
    return {"enabled": trained, "n_pairs": n, "reason": "trained model present" if trained else "needs training"}


def train(pairs: pd.DataFrame, cfg: Config) -> Path:
    if len(pairs) < cfg.correction.min_pairs:
        raise ValueError(f"need >= {cfg.correction.min_pairs} verified pairs, have {len(pairs)}")
    from datasets import Dataset
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer,
                              Seq2SeqTrainingArguments)

    t = cfg.correction.train
    tok = AutoTokenizer.from_pretrained(cfg.correction.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.correction.model)

    def enc(b):
        x = tok(b["ocr_text"], max_length=256, truncation=True)
        x["labels"] = tok(text_target=b["verified_transcription"], max_length=256, truncation=True)["input_ids"]
        return x

    ds = Dataset.from_pandas(pairs[["ocr_text", "verified_transcription"]]).map(enc, batched=True)
    out = corrector_dir(cfg)
    args = Seq2SeqTrainingArguments(str(out), num_train_epochs=t.epochs, learning_rate=t.lr,
                                    per_device_train_batch_size=t.batch_size, save_strategy="no",
                                    seed=cfg.project.seed, report_to=[])
    Seq2SeqTrainer(model=model, args=args, train_dataset=ds,
                   data_collator=DataCollatorForSeq2Seq(tok, model=model)).train()
    model.save_pretrained(out)
    tok.save_pretrained(out)
    return out


def correct(texts: list[str], cfg: Config) -> list[dict]:
    """Returns [{raw, corrected, meaning_changed, similarity}] — raw is always preserved."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from src.grading_text.embed import pair_similarity

    d = corrector_dir(cfg)
    device = get_device(cfg.project.device)
    tok = AutoTokenizer.from_pretrained(d)
    model = AutoModelForSeq2SeqLM.from_pretrained(d).to(device).eval()
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    out = tok.batch_decode(model.generate(**enc, max_new_tokens=256), skip_special_tokens=True)
    sims = pair_similarity(texts, out, cfg)
    thr = cfg.correction.meaning_change_sim_threshold
    return [{"raw": r, "corrected": c, "similarity": float(s), "meaning_changed": bool(s < thr)}
            for r, c, s in zip(texts, out, sims)]

"""Token attribution for the cross-encoder text score.

Integrated Gradients (Captum LayerIntegratedGradients on the word-embedding layer). The
baseline replaces only the *student answer* tokens with [PAD]; the question/reference
segment and special tokens stay fixed, so attributions answer "which student words moved
the score". Subword pieces are merged into words. Optional SHAP path with a text masker.
"""
from __future__ import annotations

import numpy as np

from src.common.config import Config

WORD_PREFIXES = ("▁", "Ġ")


def merge_subwords(tokens: list[str], scores: list[float]) -> list[dict]:
    """Merge subword pieces into words. A piece starting with '▁' (SentencePiece) or 'Ġ' (BPE)
    begins a new word; other pieces continue the current word. Word score = sum of pieces."""
    words: list[dict] = []
    for tok, s in zip(tokens, scores):
        starts = tok.startswith(WORD_PREFIXES)
        piece = tok.lstrip("".join(WORD_PREFIXES))
        if starts or not words:
            words.append({"word": piece, "attr": float(s)})
        else:
            words[-1]["word"] += piece
            words[-1]["attr"] += float(s)
    return [w for w in words if w["word"].strip()]


def normalize(words: list[dict]) -> list[dict]:
    total = sum(abs(w["attr"]) for w in words) or 1.0
    return [{"word": w["word"], "attr": round(w["attr"] / total, 4),
             "sign": "helped" if w["attr"] > 0 else ("hurt" if w["attr"] < 0 else "neutral")} for w in words]


def top_words(words: list[dict], k: int) -> dict[str, list[dict]]:
    pos = sorted([w for w in words if w["attr"] > 0], key=lambda w: -w["attr"])[:k]
    neg = sorted([w for w in words if w["attr"] < 0], key=lambda w: w["attr"])[:k]
    return {"helped": pos, "hurt": neg}


def _encode(ce, first: str, answer: str, max_length: int):
    enc = ce.tokenizer(first, answer, return_tensors="pt", truncation="only_first", max_length=max_length)
    seq_ids = enc.sequence_ids(0)
    student_pos = [i for i, s in enumerate(seq_ids) if s == 1]
    return enc, student_pos


def integrated_gradients(ce, first: str, answer: str, cfg: Config) -> list[dict]:
    """Per-word IG attributions (normalised to sum |attr| = 1) for the student answer."""
    import torch
    from captum.attr import LayerIntegratedGradients

    model = ce.model
    model.eval()
    dev = next(model.parameters()).device
    enc, student_pos = _encode(ce, first, answer, cfg.grading_text.cross_encoder.max_length)
    if not student_pos:
        return []
    ids = enc["input_ids"].to(dev)
    mask = enc["attention_mask"].to(dev)
    base = ids.clone()
    pad = ce.tokenizer.pad_token_id if ce.tokenizer.pad_token_id is not None else 0
    base[0, student_pos] = pad
    extra = {k: v.to(dev) for k, v in enc.items() if k == "token_type_ids"}

    def fwd(input_ids, attention_mask):
        n = input_ids.shape[0]
        kw = {k: v.expand(n, -1) for k, v in extra.items()}
        logits = model(input_ids=input_ids, attention_mask=attention_mask.expand(n, -1), **kw).logits
        return torch.sigmoid(logits[:, 0])

    lig = LayerIntegratedGradients(fwd, model.get_input_embeddings())
    attr = lig.attribute(ids, baselines=base, additional_forward_args=(mask,), n_steps=cfg.xai.ig_steps,
                         internal_batch_size=8)
    tok_attr = attr.sum(-1)[0].detach().float().cpu().numpy()
    toks = ce.tokenizer.convert_ids_to_tokens(ids[0].tolist())
    special = set(ce.tokenizer.all_special_tokens)
    keep = [i for i in student_pos if toks[i] not in special]
    return normalize(merge_subwords([toks[i] for i in keep], [tok_attr[i] for i in keep]))


def shap_attribution(ce, first: str, answer: str, cfg: Config) -> list[dict]:
    """SHAP over student-answer tokens (Text masker); reference segment held fixed."""
    import shap

    def f(answers):
        return np.asarray(ce.predict([[first, a] for a in answers], show_progress_bar=False), float)

    explainer = shap.Explainer(f, shap.maskers.Text(r"\W+"), silent=True)
    sv = explainer([answer], max_evals=cfg.xai.shap_max_evals)
    toks = [t.strip() for t in sv.data[0]]
    words = [{"word": t, "attr": float(v)} for t, v in zip(toks, sv.values[0]) if t]
    return normalize(words)


def attribute(ce, first: str, answer: str, cfg: Config) -> dict:
    words = (shap_attribution if cfg.xai.attribution == "shap" else integrated_gradients)(ce, first, answer, cfg)
    return {"method": cfg.xai.attribution, "words": words, **top_words(words, cfg.xai.top_k_tokens)}


def attribute_tokens(question: str, reference: str, answer: str, cfg: Config, ce=None) -> list[dict]:
    """Pipeline entry point: per-word attributions for the student answer, in answer order."""
    from src.grading_text.cross_encoder import first_text, get_cross_encoder

    if not (answer or "").strip():
        return []
    ce = ce or get_cross_encoder(cfg)
    first = first_text(question, reference)
    fn = shap_attribution if cfg.xai.attribution == "shap" else integrated_gradients
    return fn(ce, first, answer, cfg)

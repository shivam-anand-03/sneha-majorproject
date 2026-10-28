"""Key-concept coverage: reference key phrases -> matched / partial / missing in the student answer.

Matching tiers (first hit wins): lemma-exact, fuzzy (spelling errors), embedding (synonyms/paraphrase).
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

from src.common.config import ROOT, Config
from src.grading_text.embed import encode, get_embedder

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-']*")
STOP = {
    "the", "a", "an", "of", "to", "in", "is", "are", "and", "or", "for", "on", "by", "with", "it", "that",
    "this", "as", "be", "at", "from", "which", "its", "can", "into", "was", "were", "has", "have", "their",
}


@lru_cache(maxsize=1)
def _lemmatizer():
    import nltk

    d = str(ROOT / "data" / "cache" / "nltk")
    nltk.data.path.insert(0, d)
    for pkg in ("wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, download_dir=d, quiet=True)
    from nltk.stem import WordNetLemmatizer

    return WordNetLemmatizer()


def lemmas(text: str) -> list[str]:
    lem = _lemmatizer()
    out = []
    for w in _WORD.findall(text.lower()):
        l = lem.lemmatize(w, "n")
        out.append(lem.lemmatize(l, "v"))
    return out


def extract_concepts(reference: str, cfg: Config) -> list[str]:
    c = cfg.grading_text.concepts
    if c.extractor == "keybert":
        from keybert import KeyBERT

        kb = KeyBERT(model=get_embedder(cfg.grading_text.primary_embedder, cfg))
        kws = kb.extract_keywords(reference, keyphrase_ngram_range=tuple(c.ngram_range), stop_words="english",
                                  use_mmr=True, diversity=c.diversity, top_n=c.top_n)
        phrases = [k for k, _ in kws]
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer

        v = TfidfVectorizer(ngram_range=tuple(c.ngram_range), stop_words="english").fit([reference])
        sc = v.transform([reference]).toarray()[0]
        phrases = [v.get_feature_names_out()[i] for i in np.argsort(-sc)[: c.top_n]]
    # drop phrases fully contained in a longer selected phrase
    phrases = sorted(set(phrases), key=len, reverse=True)
    keep: list[str] = []
    for p in phrases:
        if not any(set(lemmas(p)) <= set(lemmas(k)) for k in keep):
            keep.append(p)
    return keep


def _ngrams(words: list[str], nmax: int) -> list[str]:
    return [" ".join(words[i:i + n]) for n in range(1, nmax + 1) for i in range(len(words) - n + 1)]


def match_concepts(concepts: list[str], answer: str, cfg: Config) -> list[dict]:
    from rapidfuzz import fuzz

    c = cfg.grading_text.concepts
    words = _WORD.findall(answer)
    ans_lem = lemmas(answer)
    ans_lem_set = set(ans_lem)
    grams = _ngrams(words, c.ngram_range[1]) or [answer]
    gram_emb = encode(grams, cfg) if grams else None
    con_emb = encode(concepts, cfg) if concepts else None
    rows = []
    for k, con in enumerate(concepts):
        cl = [l for l in lemmas(con) if l not in STOP] or lemmas(con)
        row = {"concept": con, "status": "missing", "match": "", "method": "", "score": 0.0}
        if cl and all(l in ans_lem_set for l in cl):
            row.update(status="matched", method="lemma", score=1.0,
                       match=" ".join(w for w in words if lemmas(w) and lemmas(w)[0] in cl))
        else:
            best_f, best_g = 0.0, ""
            for g in grams:
                if abs(len(g) - len(con)) <= max(3, len(con) // 3):
                    r = fuzz.ratio(g.lower(), con.lower())
                    if r > best_f:
                        best_f, best_g = r, g
            if best_f >= c.fuzzy_ratio:
                row.update(status="matched", method=f"fuzzy({best_f:.0f})", score=1.0, match=best_g)
            elif gram_emb is not None:
                sims = gram_emb @ con_emb[k]
                j = int(np.argmax(sims))
                s = float(sims[j])
                if s >= c.full_match:
                    row.update(status="matched", method=f"embedding({s:.2f})", score=1.0, match=grams[j])
                elif s >= c.partial_match:
                    row.update(status="partial", method=f"embedding({s:.2f})", score=0.5, match=grams[j])
                else:
                    row.update(method=f"best embedding {s:.2f}", match="")
        rows.append(row)
    return rows


def coverage(rows: list[dict]) -> float:
    return float(np.mean([r["score"] for r in rows])) if rows else 0.0

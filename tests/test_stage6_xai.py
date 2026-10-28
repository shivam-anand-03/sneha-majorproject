import random

from src.xai.attribution import merge_subwords, normalize, top_words
from src.xai.faithfulness import delete_words, deletion_test, summarize
from src.xai.justification import evidence_from_result, justify, template_justification, unsupported_numbers
from src.xai.review import needs_review

CONCEPTS = [
    {"concept": "last in first out", "status": "matched", "match": "last in first out"},
    {"concept": "push operation", "status": "partial", "match": "adds item"},
    {"concept": "pop operation", "status": "missing", "match": ""},
]


def _result(**kw):
    r = {
        "lines": [{"confidence": 0.9, "needs_review": False, "words": []}],
        "diagrams": [],
        "text_grade": {"features": {"embed_sim": 0.8, "cross_encoder": 0.72, "concept_coverage": 0.5},
                       "concept_table": CONCEPTS,
                       "token_attributions": [{"word": "LIFO", "attr": 0.3}, {"word": "maybe", "attr": -0.1}]},
        "score": {"text_score": 0.7, "diagram_score": None, "weights": {"text": 1.0, "diagram": 0.0},
                  "raw_marks": 3.5, "marks": 3.5, "total_marks": 5, "history_used": False, "history_expected": None},
        "history": None,
        "review": {"needed": False, "reasons": []},
    }
    r.update(kw)
    return r


def _ev(cfg, **kw):
    r = _result(**kw)
    r["review"] = dict(zip(("needed", "reasons"), needs_review(r, cfg)))
    return evidence_from_result(r)


def test_merge_subwords_sentencepiece_and_bpe():
    w = merge_subwords(["▁stack", "▁push", "es", "▁it", "ems"], [0.1, 0.2, 0.1, 0.05, 0.05])
    assert [x["word"] for x in w] == ["stack", "pushes", "items"]
    assert abs(w[1]["attr"] - 0.3) < 1e-9
    w2 = merge_subwords(["Ġlast", "Ġin", "put", "ted"], [1, 1, 1, 1])
    assert [x["word"] for x in w2] == ["last", "inputted"]


def test_normalize_and_top_words():
    n = normalize([{"word": "a", "attr": 2.0}, {"word": "b", "attr": -1.0}, {"word": "c", "attr": 1.0}])
    assert abs(sum(abs(x["attr"]) for x in n) - 1.0) < 1e-3
    t = top_words(n, 1)
    assert t["helped"][0]["word"] == "a" and t["hurt"][0]["word"] == "b"


def test_provenance_guard(cfg):
    ev = _ev(cfg)
    assert unsupported_numbers("Awarded 3.5 out of 5 marks; text score 0.70 (70%).", ev) == []
    assert unsupported_numbers("The student scored 4.5 out of 5.", ev) == ["4.5"]
    assert unsupported_numbers("Coverage was 93%.", ev) == ["93%"]


def test_template_mentions_concepts_and_review(cfg):
    ev = _ev(cfg, lines=[{"confidence": 0.4, "words": [], "needs_review": True}])
    txt = template_justification(ev)
    assert "pop operation" in txt and "last in first out" in txt and "push operation" in txt
    assert "LIFO" in txt and "Teacher review needed" in txt
    assert unsupported_numbers(txt, ev) == []  # template never invents numbers


def test_justify_template_engine(cfg):
    from src.common.config import load_config

    c = load_config(overrides={"xai": {"justification": {"engine": "template"}}})
    out = justify(_result(), c)
    assert out["engine"] == "template" and "3.5" in out["text"]


def test_review_flags(cfg):
    ok, reasons = needs_review(_result(), cfg)
    assert not ok and reasons == []
    tg = {"features": {"embed_sim": 0.9, "cross_encoder": 0.2}}
    flag, r = needs_review(_result(text_grade=tg), cfg)
    assert flag and "disagree" in r[0]
    flag, r = needs_review(_result(lines=[{"confidence": 0.3, "needs_review": False}]), cfg)
    assert flag and "OCR" in r[0]
    flag, r = needs_review(_result(diagrams=[{"sim_transformed": 0.9, "vlm": None, "score": 0.9}]), cfg)
    assert flag and "unavailable" in r[0]
    flag, r = needs_review(_result(diagrams=[{"sim_transformed": 0.95, "vlm": {"score": 0.3}, "score": 0.5}]), cfg)
    assert flag and "disagree" in r[0]
    flag, r = needs_review(_result(history={"anomaly": True}), cfg)
    assert flag and "history" in r[0]


def test_deletion_test_rewards_faithful_attribution():
    # toy scorer: score = fraction of key words present
    keys = {"stack", "lifo"}

    def score(_, a):
        return len(keys & {w.lower() for w in a.split()}) / len(keys)

    words = [{"word": "stack", "attr": 0.5}, {"word": "LIFO", "attr": 0.4}, {"word": "the", "attr": 0.01}]
    r = deletion_test(score, "", "the stack is LIFO and very nice indeed", words, [1, 2], random.Random(0))
    s = summarize([r], [1, 2])
    assert s["mean_drop_top2"] == 1.0 and s["mean_drop_top2"] >= s["mean_drop_random2"]
    assert delete_words("a b a", ["a"], 1) == "b a"

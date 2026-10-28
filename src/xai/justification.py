"""Natural-language justification built ONLY from structured evidence.

An LLM may phrase it, but a provenance guard rejects any output containing a number that
is not in the evidence; the deterministic template is then used. The template always works
offline and is the reference behaviour.
"""
from __future__ import annotations

import re
from typing import Any

from src.common.config import Config
from src.common.llm_client import LLMUnavailable, get_client
from src.common.prompts import JUSTIFICATION_SCHEMA, justification_prompt

_NUM = re.compile(r"(?<![A-Za-z])(\d+(?:\.\d+)?)(\s*%)?")


def _r(x: Any, nd: int = 2):
    return None if x is None else round(float(x), nd)


def summarize_ocr(lines: list[dict] | None) -> dict | None:
    if not lines:
        return None
    confs = [l["confidence"] for l in lines if l.get("confidence") is not None]
    low = [w["word"] for l in lines for w in l.get("words", []) if w.get("low")]
    return {"mean_confidence": _r(sum(confs) / len(confs)) if confs else None,
            "min_confidence": _r(min(confs)) if confs else None,
            "lines": len(lines), "lines_flagged": sum(bool(l.get("needs_review")) for l in lines),
            "low_confidence_words": low[:10]}


def _diagram_evidence(diagrams: list[dict]) -> dict | None:
    if not diagrams:
        return None
    d = max(diagrams, key=lambda x: x.get("score") or 0.0)
    vlm = d.get("vlm") or {}
    return {"similarity": _r(d.get("sim_transformed")), "vlm_score": _r(vlm.get("score")),
            "rubric": {k: {"score": _r(v.get("score")), "reason": v.get("reason", "")}
                       for k, v in (vlm.get("criteria") or {}).items()},
            "missing_elements": vlm.get("missing_elements", [])}


def _history_evidence(h: dict | None, sc: dict) -> dict | None:
    if not h:
        return None
    keep = {k: h[k] for k in ("anomaly", "weak_topics", "flags", "feedback") if k in h}
    keep["expected_marks_fraction"] = _r(sc.get("history_expected"))
    keep["used_in_score"] = bool(sc.get("history_used"))
    return keep


def evidence_from_result(result: dict) -> dict:
    sc, tg = result.get("score") or {}, result.get("text_grade") or {}
    table = tg.get("concept_table") or []
    by = {s: [{"concept": c["concept"], "student_phrase": c.get("match", "")} for c in table if c["status"] == s]
          for s in ("matched", "partial", "missing")}
    attr = tg.get("token_attributions") or []
    k = 5
    helped = [w["word"] for w in sorted(attr, key=lambda w: -w["attr"]) if w["attr"] > 0][:k]
    hurt = [w["word"] for w in sorted(attr, key=lambda w: w["attr"]) if w["attr"] < 0][:k]
    return {
        "marks": {"awarded": sc.get("marks"), "total": sc.get("total_marks"), "before_history": sc.get("raw_marks"),
                  "text_score": _r(sc.get("text_score")), "diagram_score": _r(sc.get("diagram_score")),
                  "weights": {k_: _r(v) for k_, v in (sc.get("weights") or {}).items()}},
        "text": {"features": {k_: _r(v) for k_, v in (tg.get("features") or {}).items()}, "concepts": by,
                 "helped_words": helped, "hurt_words": hurt},
        "diagram": _diagram_evidence(result.get("diagrams") or []),
        "ocr": summarize_ocr(result.get("lines")),
        "history": _history_evidence(result.get("history"), sc),
        "review": result.get("review") or {"needed": False, "reasons": []},
    }


def _evidence_numbers(obj: Any, acc: set[float]) -> set[float]:
    if isinstance(obj, bool) or obj is None:
        return acc
    if isinstance(obj, (int, float)):
        acc.add(float(obj))
    elif isinstance(obj, str):
        acc.update(float(m.group(1)) for m in _NUM.finditer(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _evidence_numbers(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _evidence_numbers(v, acc)
    return acc


def unsupported_numbers(text: str, evidence: dict) -> list[str]:
    """Numbers in text that cannot be traced to the evidence (allowing rounding and % forms)."""
    allowed = _evidence_numbers(evidence, set())
    bad = []
    for m in _NUM.finditer(text):
        raw, pct = m.group(1), bool(m.group(2))
        val = float(raw)
        nd = len(raw.split(".")[1]) if "." in raw else 0
        ok = any(round(a, nd) == val or (pct and round(a * 100, nd) == val) for a in allowed)
        if not ok:
            bad.append(m.group(0).strip())
    return bad


def template_justification(ev: dict) -> str:
    m, t = ev["marks"], ev["text"]
    parts = [f"Awarded {m['awarded']} out of {m['total']} marks."]
    w = m["weights"]
    if m.get("before_history") is not None and m["before_history"] != m["awarded"]:
        parts.append(f"(Model marks before the optional history adjustment: {m['before_history']}.)")
    if m.get("diagram_score") is not None:
        parts.append(f"The text answer scored {m['text_score']} and the diagram scored {m['diagram_score']} "
                     f"(weights: text {w.get('text')}, diagram {w.get('diagram')}).")
    else:
        parts.append(f"The text answer scored {m['text_score']} (on a 0-1 scale).")
    c = t["concepts"]
    if c["matched"]:
        parts.append("Key concepts covered: " + "; ".join(
            f"'{x['concept']}' (student wrote '{x['student_phrase']}')" for x in c["matched"]) + ".")
    if c["partial"]:
        parts.append("Partly covered: " + ", ".join(f"'{x['concept']}'" for x in c["partial"]) + ".")
    if c["missing"]:
        parts.append("Missing key concepts: " + ", ".join(f"'{x['concept']}'" for x in c["missing"]) + ".")
    if t["helped_words"]:
        parts.append("Words that raised the score most: " + ", ".join(t["helped_words"]) + ".")
    if t["hurt_words"]:
        parts.append("Words that lowered it: " + ", ".join(t["hurt_words"]) + ".")
    d = ev.get("diagram")
    if d:
        reasons = [f"{k}: {v['reason']}" for k, v in d["rubric"].items() if v.get("reason")]
        if reasons:
            parts.append("Diagram check - " + " ".join(reasons))
        if d["missing_elements"]:
            parts.append("Diagram is missing: " + ", ".join(d["missing_elements"]) + ".")
    if ev["review"]["needed"]:
        parts.append("Teacher review needed: " + "; ".join(ev["review"]["reasons"]) + ".")
    return " ".join(parts)


def justify(result: dict, cfg: Config) -> dict:
    """result: pipeline result dict (with review already set). Returns {text, engine, rejected_reason?}."""
    ev = evidence_from_result(result)
    j = cfg.xai.justification
    template = template_justification(ev)
    if j.engine != "llm":
        return {"text": template, "engine": "template"}
    client = get_client(cfg)
    strategies = ([f"gemini:{m}" for m in cfg.ocr.gemini.models[:1]] if j.backend == "gemini"
                  else [f"local-text:{j.model}"])
    try:
        res = client.structured_call(justification_prompt(ev), JUSTIFICATION_SCHEMA, strategies,
                                     max_tokens=j.max_new_tokens, purpose="justification")
    except LLMUnavailable as e:
        return {"text": template, "engine": "template", "rejected_reason": f"llm unavailable: {e}"[:200]}
    text = str(res.data["justification"]).strip()
    bad = unsupported_numbers(text, ev)
    if bad:
        return {"text": template, "engine": "template", "rejected_reason": f"unsupported numbers {bad}",
                "rejected_llm_text": text}
    if ev["review"]["needed"] and "review" not in text.lower():
        text += " Teacher review needed: " + "; ".join(ev["review"]["reasons"]) + "."
    return {"text": text, "engine": res.model}

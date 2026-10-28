"""Personalised feedback built only from facts: history profile + current concept table."""
from __future__ import annotations

from src.common.config import Config


def personalized_feedback(profile: dict, concept_table: list[dict], current_topic: str | None,
                          cfg: Config) -> list[str]:
    tips: list[str] = []
    missing = [r["concept"] for r in concept_table if r.get("status") == "missing"]
    partial = [r["concept"] for r in concept_table if r.get("status") == "partial"]
    if missing:
        tips.append("Revise these points that were missing from your answer: " + ", ".join(missing) + ".")
    if partial:
        tips.append("Explain these more precisely (only partly covered): " + ", ".join(partial) + ".")
    if profile.get("n_attempts", 0) == 0:
        return tips
    weak = profile.get("weak_topics", [])
    tm = profile.get("topic_means", {})
    if current_topic and current_topic in weak:
        tips.append(f"'{current_topic}' is a recurring weak topic (average {tm[current_topic]:.0%} "
                    f"over past attempts); prioritise it in revision.")
    others = [t for t in weak if t != current_topic]
    if others:
        tips.append("Other topics below " + f"{cfg.history.weak_topic_threshold:.0%}" + " on average: "
                    + ", ".join(f"{t} ({tm[t]:.0%})" for t in others[:3]) + ".")
    tr = profile.get("trend", 0.0)
    if tr <= -0.05:
        tips.append(f"Recent scores are trending down ({tr:+.2f} per attempt over the last "
                    f"{cfg.history.trend_window} attempts).")
    elif tr >= 0.05:
        tips.append(f"Recent scores are improving ({tr:+.2f} per attempt); keep the same approach.")
    return tips

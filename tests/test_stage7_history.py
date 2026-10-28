import pandas as pd

from src.common.config import load_config
from src.history.apply import apply_history
from src.history.feedback import personalized_feedback
from src.history.model import anomaly_check, build_training_frame, expected_score, fit_history_model
from src.history.store import student_profile, trend_slope
from src.history.synthetic import make_history


def test_default_flag_leaves_marks_unchanged(cfg):
    assert cfg.history.use_history_in_score is False
    r = apply_history(0.75, 0.3, cfg)
    assert r["marks"] == 0.75 and r["history_used"] is False and r["expected"] == 0.3 and r["model_marks"] == 0.75


def test_flag_on_blends_and_reports():
    c = load_config(overrides={"history": {"use_history_in_score": True, "history_blend": 0.2}})
    r = apply_history(1.0, 0.5, c)
    assert abs(r["marks"] - 0.9) < 1e-9 and r["history_used"]
    assert apply_history(1.0, None, c)["marks"] == 1.0  # no history -> untouched


def test_weak_topics_and_trend(cfg):
    h = pd.DataFrame({"student_id_anonymized": "S1", "topic": ["A", "A", "B", "B"], "question_id": list("wxyz"),
                      "date": ["2025-01-01", "2025-01-08", "2025-01-15", "2025-01-22"],
                      "score_norm": [0.2, 0.3, 0.9, 0.8], "attempt_no": [1, 2, 3, 4]})
    p = student_profile(h, "S1", cfg)
    assert p["weak_topics"] == ["A"] and p["n_attempts"] == 4
    assert trend_slope([0.1, 0.2, 0.3], 3) > 0


def test_anomaly_flag_only_flags(cfg):
    hist = make_history([f"S{i}" for i in range(40)], seed=1)
    hm = fit_history_model(build_training_frame(hist, cfg), cfg)
    e = expected_score(hm, hist, "S0", "E01", cfg)
    assert 0 <= e <= 1
    far = anomaly_check(1.0 if e < 0.5 else 0.0, e, hm, cfg)
    assert far["flag"] and "marks unchanged" in far["reason"]
    assert anomaly_check(e, e, hm, cfg)["flag"] is False
    assert anomaly_check(0.5, None, hm, cfg)["flag"] is False


def test_feedback_is_factual(cfg):
    prof = {"n_attempts": 5, "weak_topics": ["stacks"], "topic_means": {"stacks": 0.3}, "trend": -0.1}
    table = [{"concept": "LIFO order", "status": "missing"}, {"concept": "push", "status": "matched"}]
    fb = personalized_feedback(prof, table, "stacks", cfg)
    text = " ".join(fb)
    assert "LIFO order" in text and "push" not in text and "30%" in text and "trending down" in text
    assert personalized_feedback({"n_attempts": 0}, [], None, cfg) == []


def test_pipeline_hook(cfg, tmp_path):
    from src.history.pipeline_hook import history_for_result
    from src.history.store import COLUMNS

    p = tmp_path / "h.csv"
    make_history([f"S{i}" for i in range(30)], seed=3).to_csv(p, index=False)
    c = load_config(overrides={"paths": {"history": str(p)}})
    table = [{"concept": "LIFO", "status": "missing"}]
    r = history_for_result("S1", "E01", 0.6, table, c)
    assert 0 <= r["expected"] <= 1 and isinstance(r["anomaly"], bool) and any("LIFO" in f for f in r["feedback"])
    u = history_for_result("UNKNOWN", "E01", 0.6, table, c)
    assert u["expected"] is None and u["anomaly"] is False and u["feedback"] == [
        "Revise these points that were missing from your answer: LIFO."]
    assert set(COLUMNS) <= set(pd.read_csv(p).columns)

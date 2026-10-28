import pandas as pd
import pytest

from src.common.config import load_config
from src.data.anonymize import anon_id, anonymize_frame, scrub_text
from src.data.splits import assert_no_group_leakage, group_split


def test_config_attribute_access(cfg):
    assert cfg.pdf.dpi == 300
    assert cfg.history.use_history_in_score is False
    assert cfg.grading_diagram.transform.k == 15


def test_config_overrides():
    c = load_config(overrides={"pdf": {"dpi": 150}})
    assert c.pdf.dpi == 150 and c.pdf.backend == "pymupdf"


def test_anon_id_stable_and_salted(cfg, monkeypatch):
    a = anon_id("21CSE1234", cfg, salt="s1")
    assert a == anon_id(" 21cse1234 ", cfg, salt="s1")
    assert a != anon_id("21CSE1234", cfg, salt="s2")
    assert a.startswith(cfg.anonymization.id_prefix) and "1234" not in a


def test_scrub_text_removes_roll_numbers_and_names(cfg):
    t = scrub_text("Riya Sharma 21CSE1234 says ENR20210001 hi", cfg, names=["Riya Sharma", "Riya"])
    assert "Riya" not in t and "21CSE1234" not in t and "ENR20210001" not in t


def test_anonymize_frame(cfg):
    df = pd.DataFrame({"student_id": ["A1", "A2"], "student_name": ["Asha Rao", "Vik Das"],
                       "student_answer": ["Asha wrote this", "fine"]})
    out = anonymize_frame(df, cfg)
    assert "student_id" not in out and "student_name" not in out
    assert "Asha" not in out.student_answer[0]
    assert out.student_id_anonymized.str.startswith("STU_").all()


def test_group_split_no_leakage():
    df = pd.DataFrame({"g": [i // 5 for i in range(200)], "x": range(200)})
    s = group_split(df, "g", 0.2, seed=42, val_frac=0.1)
    assert_no_group_leakage(s, "g")
    assert sum(len(v) for v in s.values()) == 200
    s2 = group_split(df, "g", 0.2, seed=42, val_frac=0.1)
    assert s["test"].x.tolist() == s2["test"].x.tolist()  # deterministic


def test_leakage_detected():
    with pytest.raises(AssertionError):
        assert_no_group_leakage({"a": pd.DataFrame({"g": [1]}), "b": pd.DataFrame({"g": [1]})}, "g")

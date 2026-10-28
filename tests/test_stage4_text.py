import numpy as np

from src.correction.byt5 import correction_status
from src.grading_text import combiner
from src.grading_text.metrics import grading_metrics
from src.grading_text.robustness import misspell


def test_metrics_perfect_agreement():
    m = grading_metrics([0, 2.5, 5, 3], [0, 2.5, 5, 3], 5)
    assert m["qwk"] == 1.0 and m["mae"] == 0 and m["within_0.5"] == 1.0 and abs(m["pearson"] - 1) < 1e-9


def test_metrics_within_tolerance():
    m = grading_metrics([1.0, 2.0], [1.5, 3.0], 5)
    assert m["within_0.5"] == 0.5


def test_coverage_gated_by_cross_encoder(cfg):
    g = cfg.grading_text.combiner.coverage_gate_ce_min
    out = combiner.gated_coverage(np.array([1.0, 1.0]), np.array([0.05, 0.9]), g)
    assert out[0] < 0.2 and out[1] == 1.0  # keyword-stuffed (low CE) gets little coverage credit


def test_combiner_fit_predict(cfg, tmp_path):
    from src.common.config import load_config
    c = load_config(overrides={"grading_text": {"combiner": {"weights_path": str(tmp_path / "w.json")}}})
    rng = np.random.default_rng(0)
    f = {k: rng.random(200) for k in ["embed_sim", "cross_encoder", "concept_coverage"]}
    y = 0.8 * f["cross_encoder"] + 0.1 * f["embed_sim"]
    w = combiner.fit(f, y, c)
    assert w["coef"][1] > w["coef"][0] and all(x >= 0 for x in w["coef"])
    p = combiner.predict(f, c, w)
    assert p.min() >= 0 and p.max() <= 1 and np.corrcoef(p, y)[0, 1] > 0.95


def test_correction_skipped_without_pairs(cfg):
    st = correction_status(cfg)
    assert st["enabled"] is False and "skipped" in st["reason"]


def test_misspell_changes_words():
    import random
    t = "the processor executes instructions sequentially"
    assert misspell(t, random.Random(0), rate=1.0) != t

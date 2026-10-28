import numpy as np
import pytest

from src.grading_diagram import embed_sim, scorer, vlm_rubric
from src.grading_diagram.embed_sim import blend, paper_transform, prepare, transform01


def test_paper_transform_values():
    assert paper_transform(0.875, 15, 0.875) == pytest.approx(0.75)
    assert paper_transform(1.0, 15, 0.875) == pytest.approx(0.5 + 0.5 / (1 + np.exp(-1.875)))
    assert paper_transform(0.0, 15, 0.875) == pytest.approx(0.5, abs=1e-5)
    assert transform01(0.0, 15, 0.875) == pytest.approx(0.0, abs=1e-5)
    assert transform01(0.875, 15, 0.875) == pytest.approx(0.5)


def test_prepare_pads_square():
    im = prepare(np.zeros((50, 200, 3), np.uint8), 64)
    a = np.array(im)
    assert a.shape == (64, 64, 3) and a[0, 0].tolist() == [255, 255, 255]


def test_blend_weights():
    assert blend({"clip": 0.8, "dino": 0.4}, {"clip": 0.5, "dino": 0.5}) == pytest.approx(0.6)
    assert blend({"siglip": 0.7}, {"clip": 1.0}) == pytest.approx(0.7)


def _patch(monkeypatch, sims, vlm):
    monkeypatch.setattr(embed_sim, "similarities", lambda *a, **k: sims)
    monkeypatch.setattr(vlm_rubric, "rubric_check", lambda *a, **k: vlm)


def test_scorer_blends_vlm(monkeypatch, cfg):
    _patch(monkeypatch, {"clip": 0.95, "dino": 0.95},
           ({"score": 0.5, "criteria": {}, "missing_elements": ["arrow"], "raw": {}, "model": "x"}, "ok"))
    r = scorer.grade_diagram(np.ones((10, 10, 3), np.uint8), np.ones((10, 10, 3), np.uint8), "", cfg, use_vlm=True)
    st = float(transform01(0.95, 15, 0.875))
    w = cfg.grading_diagram.blend
    assert r["score"] == pytest.approx((w.w_sim * st + w.w_vlm * 0.5) / (w.w_sim + w.w_vlm))
    assert r["disagreement"] == pytest.approx(abs(st - 0.5))


def test_scorer_vlm_unavailable_flags_review(monkeypatch, cfg):
    _patch(monkeypatch, {"clip": 0.9}, (None, "VLM unavailable: no key"))
    r = scorer.grade_diagram(np.ones((10, 10, 3), np.uint8), np.ones((10, 10, 3), np.uint8), "", cfg, use_vlm=True)
    assert r["vlm"] is None and r["needs_review"] and "unavailable" in r["reasons"][0]


def test_scorer_missing_diagram(cfg):
    r = scorer.grade_diagram(None, np.ones((10, 10, 3), np.uint8), "", cfg)
    assert r["score"] == 0.0 and r["needs_review"]

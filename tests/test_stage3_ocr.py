import numpy as np

from src.ocr import router
from src.ocr.base import OCRLine
from src.ocr.metrics import cer_wer


def test_cer_wer_basic():
    m = cer_wer(["hello world"], ["hallo world"])
    assert abs(m["cer"] - 1 / 11) < 1e-6 and abs(m["wer"] - 0.5) < 1e-6


class _Fake:
    def __init__(self, name, conf, text):
        self.name, self.conf, self.text, self.calls = name, conf, text, 0

    def available(self):
        return True

    def recognize(self, crops, hard=False):
        self.calls += len(crops)
        return [OCRLine(text=self.text, engine=self.name, confidence=self.conf,
                        words=[{"word": w, "conf": self.conf, "low": (self.conf or 1) < 0.6} for w in self.text.split()],
                        candidates={self.name: self.text}) for _ in crops]


def test_low_confidence_routes_to_fallback_and_keeps_both(cfg, monkeypatch):
    prim, fb = _Fake("trocr", 0.4, "teh cat"), _Fake("vlm", None, "the cat")
    monkeypatch.setattr(router, "_ENGINES", {"trocr": prim, "vlm": fb})
    out = router.recognize_lines([np.zeros((10, 10), np.uint8)], cfg, engine="trocr", use_fallback=True)
    ln = out[0]
    assert fb.calls == 1
    assert ln.text == "teh cat"                       # primary kept as default, fallback never auto-trusted
    assert ln.candidates == {"trocr": "teh cat", "vlm": "the cat"}
    assert ln.needs_review and any("low confidence" in r for r in ln.review_reasons)


def test_high_confidence_skips_fallback(cfg, monkeypatch):
    prim, fb = _Fake("trocr", 0.95, "the cat"), _Fake("vlm", None, "x")
    monkeypatch.setattr(router, "_ENGINES", {"trocr": prim, "vlm": fb})
    out = router.recognize_lines([np.zeros((10, 10), np.uint8)], cfg, engine="trocr", use_fallback=True)
    assert fb.calls == 0 and not out[0].needs_review

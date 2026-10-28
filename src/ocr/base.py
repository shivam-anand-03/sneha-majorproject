"""Common OCR result type."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class OCRLine:
    text: str                                   # text used downstream (teacher may edit)
    engine: str
    confidence: float | None = None             # line-level probability (None if engine gives none)
    words: list[dict] = field(default_factory=list)   # {word, conf, low}
    candidates: dict[str, str] = field(default_factory=dict)  # engine -> raw text, always kept
    needs_review: bool = False
    review_reasons: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

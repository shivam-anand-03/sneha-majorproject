"""TrOCR line recognizer with token-level confidence from beam-search transition scores."""
from __future__ import annotations

import time

import numpy as np
from PIL import Image

from src.common.config import Config
from src.common.gpu import get_device, run_batched_with_oom_fallback, torch_dtype
from src.ocr.base import OCRLine


class TrOCR:
    name = "trocr"

    def __init__(self, cfg: Config, model_name: str | None = None):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self.cfg = cfg
        c = cfg.ocr.trocr
        self.model_name = model_name or c.model
        self.device = get_device(cfg.project.device)
        self.dtype = torch_dtype(self.device, cfg.project.fp16)
        self.processor = TrOCRProcessor.from_pretrained(self.model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name, torch_dtype=self.dtype).to(self.device)
        self.model.eval()

    def _words_with_conf(self, token_ids: list[int], token_probs: list[float]) -> list[dict]:
        tok = self.processor.tokenizer
        thr = self.cfg.ocr.confidence.word_low_threshold
        words: list[dict] = []
        cur, probs = "", []
        for tid, p in zip(token_ids, token_probs):
            if tid in tok.all_special_ids:
                continue
            piece = tok.convert_ids_to_tokens(tid)
            starts = piece.startswith("Ġ") or piece.startswith("▁")
            if starts and cur:
                words.append((cur, probs))
                cur, probs = "", []
            cur += tok.convert_tokens_to_string([piece]).strip() if starts else tok.convert_tokens_to_string([piece])
            probs.append(p)
        if cur:
            words.append((cur, probs))
        out = []
        for w, ps in words:
            conf = float(np.exp(np.mean(np.log(np.clip(ps, 1e-9, 1)))))  # geometric mean over tokens
            if w.strip():
                out.append({"word": w.strip(), "conf": round(conf, 4), "low": conf < thr})
        return out

    def _batch(self, crops: list[np.ndarray]) -> list[OCRLine]:
        import torch

        c = self.cfg.ocr.trocr
        t0 = time.time()
        pil = [Image.fromarray(x).convert("RGB") for x in crops]
        pv = self.processor(images=pil, return_tensors="pt").pixel_values.to(self.device, self.dtype)
        with torch.inference_mode():
            gen = self.model.generate(pv, num_beams=c.num_beams, max_new_tokens=c.max_new_tokens,
                                      output_scores=True, return_dict_in_generate=True, early_stopping=True)
            trans = self.model.compute_transition_scores(gen.sequences, gen.scores,
                                                         getattr(gen, "beam_indices", None), normalize_logits=False)
        texts = self.processor.batch_decode(gen.sequences, skip_special_tokens=True)
        dt = (time.time() - t0) / len(crops)
        out = []
        for i, text in enumerate(texts):
            seq = gen.sequences[i, 1:].tolist()  # first token is decoder start
            lp = trans[i].float().cpu().numpy()
            n = min(len(seq), len(lp))
            probs = np.exp(lp[:n]).tolist()
            ids = seq[:n]
            keep = [p for t, p in zip(ids, probs) if t not in self.processor.tokenizer.all_special_ids]
            line_conf = float(np.exp(np.mean(np.log(np.clip(keep, 1e-9, 1))))) if keep else 0.0
            out.append(OCRLine(text=text.strip(), engine=self.name, confidence=round(line_conf, 4),
                               words=self._words_with_conf(ids, probs), candidates={self.name: text.strip()},
                               seconds=dt))
        return out

    def recognize(self, crops: list[np.ndarray]) -> list[OCRLine]:
        if not crops:
            return []
        sizes = [s for s in self.cfg.gpu.oom_retry_batch_sizes if s <= self.cfg.ocr.trocr.batch_size] or [1]
        return run_batched_with_oom_fallback(self._batch, crops, sizes)

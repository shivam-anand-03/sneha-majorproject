"""The one module through which every LLM / VLM call goes (Gemini API or local Qwen).

Owns: strategy fallback order, timeouts, retry policy (429/5xx/timeouts only, exponential
backoff), strict JSON parsing + schema validation, and per-call logging
(request id, model, tokens, latency). Never logs the API key.

A *strategy* string is "gemini:<model>", "local-vl:<hf model>" or "local-text:<hf model>".
"""
from __future__ import annotations

import io
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from src.common.config import Config, resolve
from src.common.gpu import free_memory, get_device, is_oom

log = logging.getLogger(__name__)

RETRYABLE_HTTP = {408, 429, 500, 502, 503, 504}


class LLMUnavailable(RuntimeError):
    """No strategy could serve the request (no key, opt-out, all failed)."""


@dataclass
class LLMResult:
    data: dict
    model: str
    request_id: str
    latency_s: float
    tokens_in: int | None = None
    tokens_out: int | None = None
    attempts: list[dict] = field(default_factory=list)


def _to_png_bytes(img: np.ndarray | Image.Image) -> bytes:
    pil = Image.fromarray(img) if isinstance(img, np.ndarray) else img
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _validate(data: Any, schema: dict) -> dict:
    """Minimal JSON-schema check: object type + required keys (recursively for nested objects)."""
    if schema.get("type") == "object":
        if not isinstance(data, dict):
            raise ValueError(f"expected object, got {type(data).__name__}")
        for k in schema.get("required", []):
            if k not in data:
                raise ValueError(f"missing key {k!r}")
        for k, sub in schema.get("properties", {}).items():
            if k in data and sub.get("type") == "object":
                _validate(data[k], sub)
    return data


class LLMClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._gemini = None
        self._local: dict[str, tuple[Any, Any]] = {}

    # ---------------------------------------------------------------- availability
    def gemini_enabled(self) -> bool:
        e = self.cfg.external_api
        return bool(e.allow_gemini and os.environ.get(e.api_key_env))

    def _gemini_client(self):
        if self._gemini is None:
            if not self.gemini_enabled():
                raise LLMUnavailable("Gemini disabled: set external_api.allow_gemini=true and "
                                     f"{self.cfg.external_api.api_key_env}")
            from google import genai
            from google.genai import types

            self._gemini = genai.Client(
                api_key=os.environ[self.cfg.external_api.api_key_env],
                http_options=types.HttpOptions(timeout=int(self.cfg.external_api.timeout_s * 1000)),
            )
        return self._gemini

    # ---------------------------------------------------------------- logging
    def _log_call(self, rec: dict) -> None:
        path = resolve(self.cfg, self.cfg.external_api.calls_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    # ---------------------------------------------------------------- providers
    def _call_gemini(self, model: str, prompt: str, images: list, schema: dict, max_tokens: int,
                     temperature: float) -> tuple[dict, int | None, int | None]:
        from google.genai import errors, types

        client = self._gemini_client()
        parts = [types.Part.from_bytes(data=_to_png_bytes(im), mime_type="image/png") for im in images]
        conf = types.GenerateContentConfig(
            temperature=temperature, max_output_tokens=max_tokens,
            response_mime_type="application/json", response_schema=schema,
        )
        e = self.cfg.external_api
        for attempt in range(e.max_retries):
            try:
                resp = client.models.generate_content(model=model, contents=[*parts, prompt], config=conf)
                usage = resp.usage_metadata
                return (json.loads(resp.text),
                        getattr(usage, "prompt_token_count", None), getattr(usage, "candidates_token_count", None))
            except errors.APIError as err:
                if getattr(err, "code", None) not in RETRYABLE_HTTP or attempt == e.max_retries - 1:
                    raise
            except TimeoutError:
                if attempt == e.max_retries - 1:
                    raise
            time.sleep(e.backoff_base_s * (2 ** attempt))
        raise RuntimeError("unreachable")

    def _local_model(self, kind: str, name: str):
        key = f"{kind}:{name}"
        if key in self._local:
            return self._local[key]
        import torch
        from transformers import AutoProcessor, AutoTokenizer, BitsAndBytesConfig

        device = get_device(self.cfg.project.device)
        quant = None
        want4 = self.cfg.ocr.vlm.load_in_4bit if kind == "local-vl" else self.cfg.xai.justification.load_in_4bit
        if device == "cuda" and want4:
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                       bnb_4bit_quant_type="nf4")
        kw = dict(torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                  quantization_config=quant, device_map=device if device == "cuda" else None)
        if kind == "local-vl":
            from transformers import AutoModelForImageTextToText

            model = AutoModelForImageTextToText.from_pretrained(name, **kw)
            proc = AutoProcessor.from_pretrained(name, max_pixels=self.cfg.ocr.vlm.max_pixels)
        else:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(name, **kw)
            proc = AutoTokenizer.from_pretrained(name)
        model.eval()
        self._local[key] = (model, proc)
        return model, proc

    def _call_local(self, kind: str, name: str, prompt: str, images: list, schema: dict, max_tokens: int):
        import torch

        model, proc = self._local_model(kind, name)
        sys_msg = "Reply with a single JSON object matching this schema, nothing else: " + json.dumps(schema)
        if kind == "local-vl":
            pil = [Image.fromarray(i) if isinstance(i, np.ndarray) else i for i in images]
            content = [{"type": "image"} for _ in pil] + [{"type": "text", "text": prompt}]
            msgs = [{"role": "system", "content": [{"type": "text", "text": sys_msg}]},
                    {"role": "user", "content": content}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = proc(text=[text], images=pil or None, return_tensors="pt").to(model.device)
        else:
            msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = proc([text], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        gen = out[0, inputs["input_ids"].shape[1]:]
        tok = proc.tokenizer if hasattr(proc, "tokenizer") else proc
        raw = tok.decode(gen, skip_special_tokens=True).strip()
        # local models have no native JSON mode: take the outermost object, else fail (no repair calls)
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"local model returned no JSON: {raw[:120]!r}")
        return json.loads(raw[start:end + 1]), int(inputs["input_ids"].shape[1]), int(gen.shape[0])

    # ---------------------------------------------------------------- public
    def structured_call(self, prompt: str, schema: dict, strategies: list[str], images: list | None = None,
                        max_tokens: int = 256, temperature: float = 0.0, purpose: str = "") -> LLMResult:
        images = images or []
        rid = uuid.uuid4().hex[:12]
        attempts: list[dict] = []
        for strat in strategies:
            kind, _, model = strat.partition(":")
            t0 = time.time()
            try:
                if kind == "gemini":
                    data, tin, tout = self._call_gemini(model, prompt, images, schema, max_tokens, temperature)
                elif kind in ("local-vl", "local-text"):
                    data, tin, tout = self._call_local(kind, model, prompt, images, schema, max_tokens)
                else:
                    raise ValueError(f"unknown strategy {strat}")
                data = _validate(data, schema)
            except LLMUnavailable as e:
                attempts.append({"strategy": strat, "error": str(e)})
                continue
            except Exception as e:  # provider/parse failure -> next strategy
                if is_oom(e):
                    free_memory()
                attempts.append({"strategy": strat, "error": f"{type(e).__name__}: {str(e)[:200]}"})
                log.warning("LLM strategy %s failed (%s): %s", strat, rid, e)
                continue
            res = LLMResult(data, strat, rid, time.time() - t0, tin, tout, attempts)
            self._log_call({"request_id": rid, "purpose": purpose, "model": strat, "latency_s": round(res.latency_s, 3),
                            "tokens_in": tin, "tokens_out": tout, "failed_attempts": attempts,
                            "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
            return res
        self._log_call({"request_id": rid, "purpose": purpose, "model": None, "failed_attempts": attempts,
                        "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
        raise LLMUnavailable(f"all strategies failed: {attempts}")


_CLIENT: LLMClient | None = None


def get_client(cfg: Config) -> LLMClient:
    global _CLIENT
    if _CLIENT is None or _CLIENT.cfg is not cfg:
        _CLIENT = LLMClient(cfg)
    return _CLIENT

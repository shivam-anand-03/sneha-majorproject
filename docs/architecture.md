# Architecture

## Pipeline

`src/pipeline/evaluate.py:evaluate_sheet` grades one answer sheet and returns the result dict that the app
and the reports use. Each stage reads its settings from `configs/config.yaml`.

```
sheet (PDF / PNG / JPG)
  │
  ├─ src/preprocess/pdf.py        pages rendered at 300 dpi (PyMuPDF)
  ├─ src/preprocess/image.py      grayscale → shadow removal → denoise → deskew → ruled-line removal → CLAHE
  ├─ src/segmentation/layout.py   text vs diagram regions (page-relative sizes + hollow-shape test)
  ├─ src/segmentation/lines.py    text lines by projection profile
  │
  ├─ src/ocr/router.py            TrOCR with token confidence; low-confidence lines re-read by VLM
  │     trocr.py / paddle.py / vlm.py (Qwen2.5-VL 4-bit or Gemini)
  ├─ src/correction/byt5.py       optional OCR correction (only with ≥ 500 verified pairs)
  │
  ├─ src/grading_text/            SBERT similarity + DeBERTa cross-encoder + gated concept coverage
  │     → ridge combiner (models/text_combiner.json)
  ├─ src/grading_diagram/         CLIP / DINOv2 / SigLIP similarity → sigmoid, blended with a VLM rubric
  │
  ├─ marks = (w_text·text + w_diagram·diagram) · total, rounded to 0.5
  │
  ├─ src/history/                 learning history → feedback + anomaly flag (not marks, by default)
  └─ src/xai/                     IG token attribution, Grad-CAM, score breakdown,
                                  grounded justification, "teacher review needed" flags
```

If an optional module fails, its section is `null` and a review reason is recorded. The pipeline never fills in a fake value.

## Modules

| Path | Responsibility |
|---|---|
| `src/common/` | config loader (`config.py`), GPU/OOM helpers, the single LLM/VLM client (`llm_client.py`), prompts, experiment tracking |
| `src/data/` | IAM and ASAG loaders, anonymisation, annotation templates, student-disjoint splits, synthetic data |
| `src/preprocess/` | PDF rendering and image cleanup |
| `src/segmentation/` | layout split, line segmentation, segmentation metrics |
| `src/ocr/` | OCR engines behind one interface (`base.py`), routing by confidence, CER/WER |
| `src/correction/` | ByT5 OCR post-correction |
| `src/grading_text/` | embeddings, cross-encoder, concept coverage, combiner, robustness checks |
| `src/grading_diagram/` | image-embedding similarity, VLM rubric, scorer |
| `src/history/` | history store, LightGBM expected-score model, feedback |
| `src/xai/` | attribution, Grad-CAM, faithfulness metrics, justification, review rules |
| `src/pipeline/` | `page.py` (per-page analysis) and `evaluate.py` (whole sheet) |
| `app/` | FastAPI backend (`api.py`), result storage (`store.py`), demo pipeline, static web UI |
| `scripts/` | training, synthetic data and evaluation entry points |
| `notebooks/` | Colab-ready versions of setup, OCR comparison, fine-tuning and evaluation |
| `tests/` | one fast test file per stage (`test_stage1` … `test_stage8`) |

## App

- `app/api.py` serves the JSON API under `/api` and the static UI from `app/static/`.
- `app/pipeline_adapter.py` chooses the pipeline. It uses the full pipeline by default. It uses the demo pipeline
  (`app/mock_pipeline.py`) when `AES_DEMO_PIPELINE=1` is set, or when the full pipeline fails to import.
- Each evaluation is stored in `outputs/app_results/<result_id>/`. That folder holds `result.json`, the uploaded
  inputs, the answer key and every crop or heat-map image the UI shows.
- Teacher overrides are appended to `outputs/logs/overrides.jsonl`.

## Design rules

- **Config, not constants.** Every threshold, weight and model name is in `configs/config.yaml`.
- **One LLM client.** All LLM/VLM calls go through `src/common/llm_client.py`. It handles timeouts, retries,
  strict JSON parsing and logging to `outputs/logs/llm_calls.jsonl`. Student text is wrapped in untrusted-input markers.
- **Privacy by default.** External APIs are opt-in, and only crops are sent. Student IDs are salted hashes.
- **Splits.** Splits are by student or by question with seed 42, never by line.
- **History doesn't change marks** unless `history.use_history_in_score: true` (see README, *Ethics and fairness*).

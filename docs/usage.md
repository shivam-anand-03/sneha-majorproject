# Usage

## Commands

All commands go through `./dev.sh`. It sets the cache variables and uses `.venv`.

| Command | What it does |
|---|---|
| `./dev.sh setup` | First-time install (see [setup.md](setup.md)) |
| `./dev.sh` | Runs setup if `.venv` is missing, then starts the app |
| `./dev.sh start` | Starts the app with the full pipeline at http://127.0.0.1:8000 |
| `./dev.sh demo` | Starts the app with the demo pipeline (no trained models needed) |
| `./dev.sh train` | Downloads Mohler, SciEntsBank and Beetle, fine-tunes the DeBERTa cross-encoder, fits the combiner → `models/` |
| `./dev.sh synthetic` | Generates synthetic answer sheets with ground truth → `data/synthetic/` |
| `./dev.sh eval` | Runs every evaluation script → `outputs/metrics_*.json` |
| `./dev.sh test` | Fast test suite (tests marked `slow` are skipped) |

Run `./dev.sh synthetic` before `./dev.sh eval`, because the segmentation, OCR, diagram and end-to-end evaluations read `data/synthetic/`.

## Grading a sheet in the UI

1. Open http://127.0.0.1:8000.
2. Upload the student's sheet (PDF, PNG or JPG, up to 40 MB).
3. Fill in the answer key: question, reference answer, total marks, rubric (optional) and reference
   diagram (optional). Student ID and topic are optional and are used only for history-based feedback.
4. Review the result:
   - line boxes on the page
   - OCR text, with low-confidence words highlighted and other engines' readings alongside
   - score breakdown and concept coverage
   - token attribution
   - diagram heat-map and rubric
   - justification and any "teacher review needed" reasons
5. Fix OCR mistakes and press **Re-grade**, or **Override** the marks (a reason is required, and every override is logged).
6. Export the result as JSON or CSV.

## HTTP API

The API has no authentication and is bound to `127.0.0.1`. Put it behind real authentication before exposing it on a network.
Errors always use this shape: `{"error": {"code": "...", "message": "..."}}`.

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/health` | none | `{"status": "ok", "pipeline": "full" \| "demo"}` |
| POST | `/api/evaluate` | multipart: `sheet`, `reference_text`, `total_marks`, optional `question`, `rubric`, `reference_diagram`, `student_id`, `topic` | result JSON with `result_id` |
| GET | `/api/results/{id}` | none | stored result |
| GET | `/api/results/{id}/files/{name}` | none | an image referenced by the result (page, crop, heat-map) |
| POST | `/api/results/{id}/regrade` | `{"lines": [{"idx": 3, "text": "corrected line"}]}` | new result after OCR corrections |
| POST | `/api/results/{id}/override` | `{"marks": 7.5, "reason": "...", "teacher": "..."}` | result with the override recorded |
| GET | `/api/results/{id}/export?fmt=json\|csv` | none | download |

Example:

```bash
curl -F sheet=@sheet.pdf -F reference_text="Photosynthesis converts light energy..." \
     -F total_marks=10 -F question="Explain photosynthesis" \
     http://127.0.0.1:8000/api/evaluate
```

FastAPI also serves interactive docs at http://127.0.0.1:8000/docs.

## Running scripts directly

Each script is a module with its own flags. Run them through the venv:

```bash
.venv/bin/python -m scripts.make_synthetic --n-iam 30 --n-answer 60
.venv/bin/python -m scripts.train_text_grader [--skip-train] [--max-train N]
.venv/bin/python -m scripts.eval_segmentation --methods projection paddle [--debug 3] [--limit N]
.venv/bin/python -m scripts.eval_ocr --engines trocr paddle vlm [--n-iam N] [--n-vlm N] [--n-sheets N]
.venv/bin/python -m scripts.eval_diagram [--n-synth 150] [--vlm-n 40] [--device cpu]
.venv/bin/python -m scripts.eval_xai [--n 100] [--n-just 10] [--device cpu]
.venv/bin/python -m scripts.eval_history
.venv/bin/python -m scripts.eval_e2e [--limit N] [--vlm] [--variants full no_preprocess no_split oracle_ocr]
```

Run them through `./dev.sh`, or `source .envrc` first. Otherwise downloads land in your home cache instead of `./.cache`.

Tests that download models or need a GPU are marked `slow`. Run them with `.venv/bin/python -m pytest -m slow`.

## Using your own answer sheets

See [data/README.md](../data/README.md#your-own-answer-sheets). In short:

1. Put scans in `data/raw/`. This folder is git-ignored.
2. Set `AES_ANON_SALT`, then import the teacher marks CSV through `src.data.annotations.import_raw_sheet_marks`, which anonymises IDs.
3. Generate line annotation templates and correct the transcriptions.
4. Once you have ≥ 500 verified lines, ByT5 OCR correction can be trained.

## Configuration

Every setting is in `configs/config.yaml`. To use a different file, set `AES_CONFIG=path/to/config.yaml`. These are the settings most often changed:

| Key | Default | Meaning |
|---|---|---|
| `project.device` | `auto` | `auto`, `cuda` or `cpu` |
| `ocr.engine` | `trocr` | primary OCR: `trocr`, `paddle`, `vlm` or `gemini` |
| `ocr.confidence.fallback_engine` | `vlm` | engine that re-reads low-confidence lines (`vlm` local, `gemini` external) |
| `external_api.allow_gemini` | `false` | allows sending crops to Gemini (also needs `GEMINI_API_KEY`) |
| `grading.weights` | text 0.7, diagram 0.3 | final mark weights |
| `grading_diagram.blend` | sim 0.4, vlm 0.6 | diagram score blend |
| `grading_diagram.vlm_rubric.backend` | `local` | `local` (Qwen2.5-VL) or `gemini` |
| `xai.justification.engine` | `llm` | `llm` or `template` |
| `history.use_history_in_score` | `false` | keep this off (see README, *Ethics and fairness*) |
| `app.max_upload_mb` | `40` | upload size limit |

Environment variables:

| Variable | Used by |
|---|---|
| `GEMINI_API_KEY` | Gemini backend (only if `allow_gemini` is true) |
| `AES_ANON_SALT` | salt for student ID hashing |
| `AES_CONFIG` | alternative config file |
| `AES_DEMO_PIPELINE=1` | force the demo pipeline (`./dev.sh demo` sets it) |
| `PORT` | app port for `./dev.sh start` |
| `TORCH_INDEX` | `cu124` or `cpu`, overrides GPU detection in `./dev.sh setup` |

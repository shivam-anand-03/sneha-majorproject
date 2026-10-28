# Multimodal Explainable Automated Evaluation of Handwritten Answer Sheets

Grades handwritten answer sheets (text + diagrams) and explains every score. It builds on Kamble et al.,
*"Multi-Modal Automated Evaluation of Handwritten Answer Sheets: SBERT + CLIP"* (SN Computer Science, 2025)
and fixes that paper's known weaknesses: Tesseract on handwriting, fixed-pixel diagram filters,
cosine-only grading that keyword stuffing can game, and CLIP's blindness to labels and arrows.

```
PDF/image ─► 300 dpi pages ─► preprocess (gray, shadow, denoise, deskew, rule removal, CLAHE)
          ─► layout: text vs diagram (page-relative + hollow-shape test)
          ─► line segmentation (projection profile)
          ─► OCR: TrOCR + token confidence ─► low-confidence lines re-read by VLM (Qwen2.5-VL / Gemini), both kept
          ─► text grade = ridge( SBERT sim, fine-tuned DeBERTa cross-encoder, gated concept coverage )
          ─► diagram grade = blend( CLIP/DINOv2/SigLIP sim → paper sigmoid, VLM rubric JSON )
          ─► marks = (w_t·text + w_d·diagram) · total            (history never changes marks by default)
          ─► XAI: IG token attribution, concept table, Grad-CAM, score breakdown, grounded justification, review flag
```

## Quick start

After cloning, one command sets everything up. It installs uv if needed, Python 3.11, CUDA or CPU torch, deps and fonts, then runs the tests:

```bash
git clone https://github.com/shivam-anand-03/sneha-majorproject.git && cd sneha-majorproject && ./dev.sh setup
```

Docs: [setup and troubleshooting](docs/setup.md) · [usage, API and config](docs/usage.md) · [architecture](docs/architecture.md)

```bash
./dev.sh setup     # first run: uv + .venv (Python 3.11) + torch (CUDA if GPU) + deps + fonts + tests; caches in ./.cache
./dev.sh train     # fine-tune cross-encoder on Mohler+SciEntsBank+Beetle, fit combiner (~15 min on RTX 4050)
./dev.sh           # start the app → http://127.0.0.1:8000
./dev.sh demo      # app with the demo pipeline (no trained models needed)
./dev.sh test      # 52 fast tests
./dev.sh synthetic && ./dev.sh eval   # regenerate data and run every evaluation
```

Gemini (optional OCR / diagram / justification backend): `export GEMINI_API_KEY=...` and set
`external_api.allow_gemini: true` in `configs/config.yaml`. This sends **line and diagram crops only**
(never the full page or name header) to Google. It is opt-in for privacy reasons.

## App

FastAPI backend plus a hand-built web UI (gold on warm white, Inter; design notes in `app/design.md`).
Upload a sheet and an answer key (question, reference answer, total marks, rubric, reference diagram). The UI shows:
- detected line boxes and crops
- editable OCR text, with low-confidence words highlighted and alternative engine readings side by side
- the score breakdown and the concept coverage table
- token attribution
- the diagram panel: reference, heat-map and rubric sub-scores
- the justification and a "teacher review needed" banner

Teachers can re-grade after correcting OCR text, override marks (logged to `outputs/logs/overrides.jsonl`)
and export JSON or CSV. Streamlit was replaced by a custom UI because Streamlit can't be styled to this standard.

## Results (measured)

All numbers come from runs in this repo (`outputs/metrics_*.json`, `outputs/logs/experiments.jsonl`).
The sheets are **synthetic** (see `data/README.md`): real IAM handwriting and font-rendered answers on
generated ruled or plain pages with skew, shadow and noise.

**Line segmentation and layout** (90 sheets, IoU ≥ 0.5)

| Method | Line recall | Line precision | Ruled-paper recall | Diagram recall | False diagrams / sheet |
|---|---|---|---|---|---|
| Projection profile + preprocessing (**chosen**) | **0.923** | **0.951** | 0.906 | 0.970 | 0.02 |
| PaddleOCR DBNet + preprocessing | 0.837 | 0.835 | 0.823 | 0.970 | 0.02 |
| Projection, no preprocessing | 0.177 | 0.296 | 0.000 | 0.485 | 0.62 |

Deskew error is 0.046° on average. Preprocessing (deskew + rule removal) is the single biggest factor.

**Text grading.** The fine-tuned DeBERTa-v3 cross-encoder scores **Pearson 0.725 / Spearman 0.734**
against human scores on the held-out ASAG validation split (student- and question-disjoint). Learned combiner weights:
cross-encoder 0.72, SBERT similarity 0.16, gated concept coverage 0.03, intercept 0.06. Coverage counts only
when the cross-encoder agrees, so keyword lists can't earn marks on their own.

**Diagram grading** (183 student/reference pairs with rule-based marks)

| Signal | Pearson vs mark |
|---|---|
| CLIP ViT-L/14 | 0.155 |
| DINOv2 | 0.037 |
| CLIP + DINOv2 + SigLIP | 0.134 |

Diagram extraction: mean IoU 0.80, recall 0.97. **Image embeddings barely see diagram meaning.** A diagram
with a missing box scores the same as a perfect one (CLIP 0.88 vs 0.87). This confirms the paper's weakness
and is why the VLM rubric check (components, labels, arrows, structure) carries 60% of the diagram score.

**History model** (synthetic histories): LightGBM predicts expected score with MAE 0.126 (mean baseline 0.150).
History is **off** in marks by default. Blending it in hurts low-ability and improving students
(MAE on atypical answers 0.51 → 0.77 at blend 0.3); see the fairness note below.

**Not measured yet.** These evaluations were stopped to finish quickly, so no numbers are claimed:
- OCR CER/WER (TrOCR vs PP-OCRv5 vs VLM)
- QWK / MAE on the ASAG test sets
- end-to-end agreement and ablations
- the VLM diagram ablation
- the saved XAI faithfulness metrics

Run `./dev.sh eval` to produce them. For reference only: the TrOCR paper reports about 3–4% CER on IAM. Expect
noticeably worse on Indian answer sheets.

## Repository

```
src/preprocess  segmentation  ocr  correction  grading_text  grading_diagram  history  xai  pipeline  common
app/            FastAPI (api.py) + static UI      scripts/   training + evaluation entry points
configs/config.yaml   every threshold, weight and model name      notebooks/  Colab-ready 01–04
tests/          52 fast tests                       data/README.md  datasets, anonymisation, own-data workflow
```

Engineering notes:
- Every LLM/VLM call goes through `src/common/llm_client.py`, which handles timeouts, retries, strict JSON and call logging.
- Student text in prompts is wrapped in untrusted-input markers.
- The justification generator is checked by a provenance guard: any number not in the evidence makes it fall back to a template.
- ByT5 OCR correction switches on only with ≥ 500 verified pairs. Raw and corrected text are always kept side by side.
- Splits are by student or question, never by line, with seed 42.

## Limitations

- **Domain shift:** TrOCR and the evaluation sheets come from IAM (Western, adult, single-column handwriting)
  and handwriting fonts. Indian exam booklets add mixed scripts, margins, strike-throughs, very poor scans and
  different letterforms. Expect higher CER until TrOCR is fine-tuned on locally verified lines.
- **Synthetic evaluation:** sheets, diagram defects and diagram marks are generated. Real teacher-marked sheets are needed for the true accuracy.
- **Diagrams:** overlapping text and drawings, diagrams drawn on top of ruled lines, and diagrams split across pages
  are only partly handled. The VLM rubric can hallucinate, which is why it is always shown with its reasons and flagged for review on disagreement.
- **Cross-encoder:** trained on English science and CS short answers. Long or essay answers and other subjects are out of distribution.
- **Hardware:** 6 GB GPU, so Qwen2.5-VL-3B runs in 4-bit instead of 7B.

## Ethics and fairness

- Student IDs are salted SHA-256 hashes. Names and roll numbers are scrubbed before storage. Real student data is never committed.
- The system is decision support. Low OCR confidence, model disagreement or history anomalies raise a
  "teacher review needed" flag, and teachers can override any score (every override is logged).
- **Learning history** feeds personalised feedback and review routing only. `history.use_history_in_score` is
  `false` by default. Turning it on pulls students towards their past, penalising those who improve, and the
  ablation shows the harm falls on the lowest-ability group. If it is ever enabled, disclose it to students,
  keep the blend ≤ 0.1 and monitor per-group error.

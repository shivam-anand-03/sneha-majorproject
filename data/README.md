# Data

Nothing in this folder except this README is committed. Real student data must never be committed.

## Public data (downloaded automatically from the Hugging Face Hub)

| Purpose | Dataset | Hub id | Notes |
|---|---|---|---|
| Handwriting OCR eval | IAM lines | `Teklia/IAM-line` | official writer-disjoint splits; only `test` is used for evaluation |
| ASAG training/eval | Mohler (CS) | `nkazi/MohlerASAG` (`cleaned/open_ended`) | 0–5 marks, two graders averaged; split **by student** (`A##` id) 70/10/20 |
| ASAG training/eval | SciEntsBank | `nkazi/SciEntsBank` | 5-way labels mapped to 0/0.5/1 (`configs/config.yaml`); official `test_ua` / `test_uq` |
| ASAG training/eval | Beetle | `nkazi/Beetle` | same mapping |
| (not used) | ASAP-SAS | `nlpatunt/D_ASAP-SAS` | rubric only, no reference answers → no (reference, answer) pairs |

## Synthetic answer sheets (`data/synthetic/`, generated)

`python -m scripts.make_synthetic --n-iam 30 --n-answer 60`

* **iam sheets** – real IAM *test* handwriting lines pasted on ruled/plain paper.
* **answer sheets** – Mohler *test* answers rendered in open-licence handwriting fonts (`data/fonts/`, Google Fonts OFL/Apache) with per-word jitter, plus a procedurally drawn diagram with controlled defects (missing box / label / arrow, reversed arrow, shuffled labels) and a rule-based mark.
* Degradations: skew (±6°), shadows, sensor noise, ruled lines + margin line.
* `manifest.json` stores ground-truth line boxes + transcriptions, diagram box, reference diagram, teacher text score and diagram mark.

Synthetic sheets are cleaner than real scans and font handwriting is easier than real handwriting; numbers on them are an upper bound for real Indian answer sheets (see README → Limitations).

## Your own answer sheets

1. Put scans/PDFs under `data/raw/` (git-ignored).
2. Teacher marks CSV with real ids → anonymise on import:
   ```python
   from src.common.config import load_config
   from src.data.annotations import import_raw_sheet_marks
   import_raw_sheet_marks("data/raw/marks.csv", load_config(), "data/annotations/sheets.csv")
   ```
   Required columns: `student_id, sheet_id, question_id, teacher_marks, teacher_diagram_marks, max_marks` (+ optional `student_name`, dropped). Set a secret salt first: `export AES_ANON_SALT=...` (never commit it).
3. Line annotation: run the pipeline once, then `src.data.annotations.create_line_template(...)` writes
   `image_path, sheet_id, student_id_anonymized, line_idx, x0..y1, ocr_text, verified_transcription` with the OCR text pre-filled; correct `verified_transcription` by hand.
4. Splits are always **by student** (`split_lines_by_student`, seed from config) – never random line-level (writer leakage).
5. Once ≥ 500 verified lines exist, ByT5 OCR correction can be trained (`src/correction/byt5.py`); below that it is skipped automatically.

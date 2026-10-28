"""FastAPI backend for the teacher app.

Run: .venv/bin/uvicorn app.api:app --host 127.0.0.1 --port 8000

Public routes (no auth): this is a single-teacher local tool bound to loopback
(config app.api_host). Put it behind real authentication before exposing it on a network.
Errors use one envelope: {"error": {"code": str, "message": str}}.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import mock_pipeline, pipeline_adapter, store
from src.common.config import load_config, resolve
from src.data.anonymize import anon_id

log = logging.getLogger("app.api")
cfg = load_config()
app = FastAPI(title="Answer sheet evaluation", version="1.0")
STATIC = Path(__file__).parent / "static"

SHEET_TYPES = {".pdf": b"%PDF", ".png": b"\x89PNG", ".jpg": b"\xff\xd8\xff", ".jpeg": b"\xff\xd8\xff"}
IMAGE_TYPES = {k: v for k, v in SHEET_TYPES.items() if k != ".pdf"}


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status, self.code, self.message = status, code, message


def _err(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


@app.exception_handler(ApiError)
async def _api_error(_: Request, e: ApiError):
    return _err(e.status, e.code, e.message)


@app.exception_handler(store.NotFound)
async def _not_found(_: Request, e: store.NotFound):
    return _err(404, "not_found", "Result or file not found")


@app.exception_handler(RequestValidationError)
async def _validation(_: Request, e: RequestValidationError):
    first = e.errors()[0] if e.errors() else {}
    field = ".".join(str(x) for x in first.get("loc", [])[1:])
    return _err(400, "invalid_request", f"{field}: {first.get('msg', 'invalid')}".strip(": "))


@app.exception_handler(mock_pipeline.BadImage)
async def _bad_image(_: Request, e: Exception):
    return _err(400, "unreadable_image", str(e))


@app.exception_handler(HTTPException)
async def _http(_: Request, e: HTTPException):
    return _err(e.status_code, "http_error", str(e.detail))


@app.exception_handler(Exception)
async def _internal(_: Request, e: Exception):
    log.exception("unhandled error")
    return _err(500, "internal_error", "The evaluation failed. Check the server log.")


def _save_upload(up: UploadFile, dest_dir: Path, stem: str, allowed: dict[str, bytes]) -> Path:
    ext = Path(up.filename or "").suffix.lower()
    if ext not in allowed:
        raise ApiError(400, "unsupported_file", f"{up.filename}: allowed types are {', '.join(sorted(allowed))}")
    limit = int(cfg.app.max_upload_mb * 1024 * 1024)
    dest = dest_dir / f"{stem}{ext}"
    size = 0
    with open(dest, "wb") as f:
        while chunk := up.file.read(1 << 20):
            size += len(chunk)
            if size > limit:
                raise ApiError(413, "file_too_large", f"{up.filename} is larger than {cfg.app.max_upload_mb} MB")
            f.write(chunk)
    with open(dest, "rb") as f:
        if not f.read(8).startswith(allowed[ext]):
            raise ApiError(400, "unsupported_file", f"{up.filename} is not a valid {ext[1:].upper()} file")
    return dest


def _display_name(filename: str | None) -> str:
    """Uploaded file stem for display only (the file itself is stored under a fixed name)."""
    stem = Path(filename or "sheet").stem
    return "".join(c for c in stem if c.isalnum() or c in "-_. ")[:80] or "sheet"


def _finish(rid: str, d: Path, result: dict) -> dict:
    result["result_id"] = rid
    store.save_result(d, result)
    return result


@app.get("/api/health")
def health():
    return {"status": "ok", "pipeline": pipeline_adapter.PIPELINE_KIND}


@app.post("/api/evaluate")
def evaluate(sheet: UploadFile = File(...), question: str = Form(""), reference_text: str = Form(...),
             total_marks: float = Form(..., gt=0, le=1000), rubric: str = Form(""),
             reference_diagram: UploadFile | None = File(None), student_id: str = Form(""), topic: str = Form("")):
    if not reference_text.strip():
        raise ApiError(400, "invalid_request", "reference_text: the model answer is required")
    rid, d = store.new_result_dir(cfg)
    try:
        sheet_path = _save_upload(sheet, d, "input_sheet", SHEET_TYPES)
        key = {"question": question, "reference_text": reference_text, "total_marks": total_marks, "rubric": rubric}
        if reference_diagram is not None and reference_diagram.filename:
            key["reference_diagram_path"] = str(_save_upload(reference_diagram, d, "input_ref_diagram", IMAGE_TYPES))
        sid = anon_id(student_id, cfg) if student_id.strip() else None
        stored = {**key, "student_id_anon": sid, "topic": topic or None}
        if "reference_diagram_path" in key:  # stored relative so a moved outputs/ folder still works
            stored["reference_diagram_path"] = Path(key["reference_diagram_path"]).name
        (d / "answer_key.json").write_text(json.dumps(stored))
        result = pipeline_adapter.run(sheet_path, key, cfg, d, sid, topic or None)
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise
    result["student_id_anon"] = sid
    result["sheet_id"] = _display_name(sheet.filename)
    return _finish(rid, d, result)


@app.get("/api/results/{rid}")
def get_result(rid: str):
    return store.load_result(cfg, rid)


@app.get("/api/results/{rid}/files/{name}")
def get_file(rid: str, name: str):
    return FileResponse(store.result_file(cfg, rid, name))


class EditedLine(BaseModel):
    idx: int = Field(ge=0)
    text: str = Field(max_length=2000)


class RegradeBody(BaseModel):
    lines: list[EditedLine]


def _answer_key(d: Path) -> tuple[dict, dict]:
    meta = json.loads((d / "answer_key.json").read_text())
    key = {k: meta[k] for k in ("question", "reference_text", "total_marks", "rubric") if k in meta}
    if meta.get("reference_diagram_path"):
        key["reference_diagram_path"] = str(d / Path(meta["reference_diagram_path"]).name)
    return key, meta


@app.post("/api/results/{rid}/regrade")
def regrade(rid: str, body: RegradeBody):
    d = store.result_dir(cfg, rid)
    with store.result_lock(rid):
        prior = store.load_result(cfg, rid)
        n = len(prior.get("lines", []))
        idxs = [e.idx for e in body.lines]
        if any(i >= n for i in idxs) or len(set(idxs)) != len(idxs):
            raise ApiError(400, "invalid_request", f"lines: each idx must be unique and below {n}")
        changed = [e.model_dump() for e in body.lines if e.text != prior["lines"][e.idx]["text"]]
        if not changed:
            raise ApiError(400, "invalid_request", "lines: no line text differs from the current transcription")
        key, meta = _answer_key(d)
        sheet_id, old_override = prior.get("sheet_id"), prior.get("override")
        result = pipeline_adapter.run(None, key, cfg, d, meta.get("student_id_anon"), meta.get("topic"),
                                      edited_lines=changed, prior=prior)
        result["student_id_anon"] = meta.get("student_id_anon")
        result["sheet_id"] = sheet_id
        # the transcription changed, so a previous override is superseded; keep it on record
        result["override"] = None
        result["superseded_override"] = old_override
        return _finish(rid, d, result)


class OverrideBody(BaseModel):
    marks: float = Field(ge=0, allow_inf_nan=False)
    reason: str = Field(min_length=3, max_length=1000)
    teacher: str = Field(min_length=1, max_length=120)


@app.post("/api/results/{rid}/override")
def override(rid: str, body: OverrideBody):
    d = store.result_dir(cfg, rid)
    with store.result_lock(rid):
        return _record_override(rid, d, body)


def _record_override(rid: str, d: Path, body: OverrideBody) -> dict:
    result = store.load_result(cfg, rid)
    total = float(result["score"]["total_marks"])
    if body.marks > total:
        raise ApiError(400, "invalid_request", f"marks: cannot exceed the total of {total:g}")
    entry = {"result_id": rid, "sheet_id": result.get("sheet_id"), "student_id_anon": result.get("student_id_anon"),
             "model_marks": result["score"]["marks"], "override_marks": body.marks, "reason": body.reason,
             "teacher": body.teacher, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    log_path = resolve(cfg, cfg.paths.overrides_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    result["override"] = {"marks": body.marks, "reason": body.reason, "teacher": body.teacher, "at": entry["at"]}
    store.save_result(d, result)
    return result


_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def _cell(v) -> object:
    """Neutralise spreadsheet formulas in free text (OCR lines, reasons, names)."""
    return "'" + v if isinstance(v, str) and v.startswith(_FORMULA_START) else v


def _csv(result: dict) -> str:
    s, tg, ov = result["score"], result["text_grade"], result.get("override") or {}
    buf = io.StringIO()
    rows: list[list] = []
    rows.append(["section", "field", "value"])
    for k in ("result_id", "sheet_id", "student_id_anon", "created_at"):
        rows.append(["sheet", k, result.get(k)])
    for k in ("text_score", "diagram_score", "raw_marks", "marks", "total_marks", "history_used"):
        rows.append(["score", k, s.get(k)])
    rows.append(["score", "final_marks", ov.get("marks", s["marks"])])
    rows.append(["score", "override_reason", ov.get("reason", "")])
    for k, v in tg.get("features", {}).items():
        rows.append(["text_features", k, v])
    for r in tg.get("concept_table", []):
        rows.append(["concept", r["concept"], f"{r['status']}: {r.get('match', '')}"])
    for ln in result.get("lines", []):
        rows.append(["ocr_line", ln["idx"] + 1, ln["text"]])
    rows.append(["review", "needed", result["review"]["needed"]])
    rows.append(["review", "reasons", " | ".join(result["review"]["reasons"])])
    rows.append(["justification", result["justification"].get("engine", ""), result["justification"]["text"]])
    csv.writer(buf).writerows([[_cell(c) for c in r] for r in rows])
    return buf.getvalue()


@app.get("/api/results/{rid}/export")
def export(rid: str, fmt: str = "json"):
    result = store.load_result(cfg, rid)
    if fmt == "csv":
        return Response(_csv(result), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="report_{rid}.csv"'})
    if fmt != "json":
        raise ApiError(400, "invalid_request", "fmt: must be json or csv")
    return Response(json.dumps(result, indent=1, default=str), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="report_{rid}.json"'})


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"], include_in_schema=False)
def api_not_found(path: str):
    """Unknown /api paths (incl. decoded '../' segments) get the error envelope, never the static app."""
    raise ApiError(404, "not_found", "No such API route")


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")

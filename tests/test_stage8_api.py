"""API tests against the demo pipeline (real segmentation, placeholder OCR/grading). Fast, CPU only."""
import io
import json
import os

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

os.environ["AES_DEMO_PIPELINE"] = "1"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from fastapi.testclient import TestClient

    from app import api

    out = tmp_path_factory.mktemp("out")
    api.cfg.project.output_dir = str(out)
    api.cfg.paths.overrides_log = str(out / "overrides.jsonl")
    return TestClient(api.app)


def _sheet_png() -> bytes:
    from src.common.config import load_config
    from src.data.synthetic import font_paths

    img = Image.new("RGB", (1240, 1754), (248, 246, 240))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_paths(load_config())[0], 40)
    for i, t in enumerate(["A prototype simulates the behaviour", "of portions of the desired software"]):
        d.text((100, 200 + 80 * i), t, font=font, fill=(20, 20, 80))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


FORM = {"question": "Role of a prototype?", "reference_text": "To simulate the behaviour of portions of the desired software product.",
        "total_marks": "5", "student_id": "21CSE1234", "topic": "SE"}


@pytest.fixture(scope="module")
def result(client):
    r = client.post("/api/evaluate", data=FORM, files={"sheet": ("s.png", _sheet_png(), "image/png")})
    assert r.status_code == 200, r.text
    return r.json()


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok", "pipeline": "demo"}


def test_evaluate_contract(result):
    for k in ("result_id", "lines", "diagrams", "text_grade", "score", "review", "justification", "page_images"):
        assert k in result
    assert len(result["lines"]) == 2
    assert result["student_id_anon"].startswith("STU_") and "1234" not in result["student_id_anon"]
    assert 0 <= result["score"]["marks"] <= 5
    assert result["config_snapshot"]["pipeline"] == "demo"


def test_raw_student_id_not_stored(client, result):
    from app import api

    d = api.store.result_dir(api.cfg, result["result_id"])
    for f in d.glob("*.json"):
        assert "21CSE1234" not in f.read_text()


def test_files_served_and_path_traversal_blocked(client, result):
    rid = result["result_id"]
    assert client.get(f"/api/results/{rid}/files/{result['lines'][0]['crop']}").status_code == 200
    for bad in ["..%2Fresult.json", "result.json", "answer_key.json", ".hidden"]:
        r = client.get(f"/api/results/{rid}/files/{bad}")
        assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    assert client.get("/api/results/nothex/files/x.png").status_code == 404


def test_regrade_uses_edited_text(client, result):
    rid = result["result_id"]
    r = client.post(f"/api/results/{rid}/regrade", json={"lines": [{"idx": 0, "text": "completely unrelated words"}]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["lines"][0]["text"] == "completely unrelated words"
    assert body["lines"][0]["candidates"]["teacher"] == "completely unrelated words"
    assert client.post(f"/api/results/{rid}/regrade", json={"lines": [{"idx": 99, "text": "x"}]}).status_code == 400


def test_override_logged(client, result):
    from app import api

    rid = result["result_id"]
    r = client.post(f"/api/results/{rid}/override", json={"marks": 4, "reason": "diagram on next page", "teacher": "T1"})
    assert r.status_code == 200 and r.json()["override"]["marks"] == 4
    log = [json.loads(l) for l in open(api.cfg.paths.overrides_log)]
    assert log[-1]["override_marks"] == 4 and "model_marks" in log[-1]
    bad = client.post(f"/api/results/{rid}/override", json={"marks": 50, "reason": "too many", "teacher": "T1"})
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "invalid_request"


def test_export(client, result):
    rid = result["result_id"]
    c = client.get(f"/api/results/{rid}/export?fmt=csv")
    assert c.status_code == 200 and "final_marks" in c.text
    j = client.get(f"/api/results/{rid}/export?fmt=json")
    assert j.json()["result_id"] == rid
    assert client.get(f"/api/results/{rid}/export?fmt=xml").status_code == 400


def test_rejects_bad_uploads(client):
    r = client.post("/api/evaluate", data=FORM, files={"sheet": ("s.exe", b"MZ..", "application/octet-stream")})
    assert r.status_code == 400 and r.json()["error"]["code"] == "unsupported_file"
    r = client.post("/api/evaluate", data=FORM, files={"sheet": ("s.png", b"not a png", "image/png")})
    assert r.status_code == 400
    r = client.post("/api/evaluate", data={**FORM, "total_marks": "-1"}, files={"sheet": ("s.png", _sheet_png(), "image/png")})
    assert r.status_code == 400 and "total_marks" in r.json()["error"]["message"]


def test_static_index(client):
    r = client.get("/")
    assert r.status_code == 200 and "Answer Sheet Evaluation" in r.text


def test_review_regressions(client, result):
    rid = result["result_id"]
    # NaN marks rejected instead of corrupting result.json
    r = client.post(f"/api/results/{rid}/override", content='{"marks": NaN, "reason": "nan test", "teacher": "T"}',
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 400
    assert client.get(f"/api/results/{rid}").status_code == 200
    # empty / duplicate / unchanged regrades are rejected and do not clear an override
    client.post(f"/api/results/{rid}/override", json={"marks": 3, "reason": "keep me", "teacher": "T"})
    cur = client.get(f"/api/results/{rid}").json()
    assert client.post(f"/api/results/{rid}/regrade", json={"lines": []}).status_code == 400
    same = {"lines": [{"idx": 1, "text": cur["lines"][1]["text"]}]}
    assert client.post(f"/api/results/{rid}/regrade", json=same).status_code == 400
    dup = {"lines": [{"idx": 1, "text": "a"}, {"idx": 1, "text": "b"}]}
    assert client.post(f"/api/results/{rid}/regrade", json=dup).status_code == 400
    assert client.get(f"/api/results/{rid}").json()["override"]["marks"] == 3
    # a real regrade keeps the display sheet id and records the superseded override
    r = client.post(f"/api/results/{rid}/regrade", json={"lines": [{"idx": 1, "text": "=1+1 formula"}]}).json()
    assert r["sheet_id"] == "s" and r["override"] is None and r["superseded_override"]["marks"] == 3
    # CSV neutralises formulas
    assert "'=1+1 formula" in client.get(f"/api/results/{rid}/export?fmt=csv").text


def test_corrupt_reference_diagram(client):
    files = {"sheet": ("s.png", _sheet_png(), "image/png"),
             "reference_diagram": ("d.jpg", b"\xff\xd8\xff\xe0truncated", "image/jpeg")}
    r = client.post("/api/evaluate", data=FORM, files=files)
    assert r.status_code == 400 and r.json()["error"]["code"] == "unreadable_image"

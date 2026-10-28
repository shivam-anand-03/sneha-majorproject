"""Synthetic answer sheets with exact ground truth (line boxes, transcriptions, diagram boxes, marks).

Sheet kinds
  iam    : real IAM handwriting test lines pasted on a page (segmentation + OCR evaluation)
  answer : a public ASAG answer rendered in a handwriting font with per-word jitter,
           optionally with a hand-drawn diagram (end-to-end grading evaluation)

Degradations (skew, shadow, noise, ruled paper) are applied after composition. Ground-truth
boxes are stored in the *unskewed* frame: deskewing should bring the page back to it.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.common.config import Config, resolve
from src.grading_diagram import synth as dsynth

INKS = [(15, 20, 80), (20, 20, 25), (30, 40, 120)]


@dataclass
class SheetGT:
    sheet_id: str
    kind: str
    student_id: str
    image_path: str
    lines: list[dict] = field(default_factory=list)       # {bbox, text}
    diagram: dict | None = None                          # {bbox, ref_path, spec, mark}
    answer_text: str = ""
    question: str = ""
    reference: str = ""
    teacher_text_score: float | None = None              # 0..1
    max_marks: float = 10.0
    degradations: dict = field(default_factory=dict)


def _paper(W: int, H: int, rng: random.Random, ruled: bool, spacing: int, margin: int) -> Image.Image:
    base = rng.randint(238, 252)
    arr = np.full((H, W, 3), (base, base, base - rng.randint(0, 8)), np.uint8)
    img = Image.fromarray(arr)
    if ruled:
        d = ImageDraw.Draw(img)
        col = (170, 190, 225)
        for y in range(margin + spacing, H - margin // 2, spacing):
            d.line([(0, y), (W, y)], fill=col, width=3)
        d.line([(int(margin * 0.8), 0), (int(margin * 0.8), H)], fill=(225, 150, 150), width=3)
    return img


def _render_hand_line(d: ImageDraw.ImageDraw, text: str, x: int, baseline: int, font: ImageFont.FreeTypeFont,
                      rng: random.Random, ink, max_w: int) -> tuple[tuple[int, int, int, int], str]:
    """Draw words with jittered baseline/size; returns tight bbox of drawn ink and text actually drawn."""
    cx = x
    drawn = []
    x0 = y0 = 10 ** 9
    x1 = y1 = -1
    space = font.getlength(" ")
    for word in text.split():
        w = font.getlength(word)
        if cx + w > x + max_w and drawn:
            break
        dy = rng.uniform(-0.06, 0.06) * font.size
        bb = d.textbbox((cx, baseline + dy), word, font=font, anchor="ls")
        d.text((cx, baseline + dy), word, font=font, fill=ink, anchor="ls")
        x0, y0, x1, y1 = min(x0, bb[0]), min(y0, bb[1]), max(x1, bb[2]), max(y1, bb[3])
        drawn.append(word)
        cx += w + space * rng.uniform(0.8, 1.6)
    return (int(x0), int(y0), int(x1), int(y1)), " ".join(drawn)


def _wrap_and_render(img: Image.Image, text: str, font_path: str, size: int, rng: random.Random,
                     x: int, y: int, max_w: int, spacing: int, ink) -> tuple[list[dict], int]:
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, size)
    words = text.split()
    lines = []
    baseline = y
    while words:
        bbox, drawn = _render_hand_line(d, " ".join(words), x + rng.randint(0, 12), baseline, font, rng, ink, max_w)
        n = len(drawn.split())
        if n == 0:
            break
        lines.append({"bbox": list(bbox), "text": drawn})
        words = words[n:]
        baseline += spacing
    return lines, baseline


def _degrade(img: np.ndarray, rng: random.Random, cfg: Config, skew: float, shadow: bool) -> np.ndarray:
    s = cfg.synthetic
    out = img.astype(np.float32)
    H, W = out.shape[:2]
    if shadow:
        gx = np.linspace(rng.uniform(0.55, 0.8), 1.0, W)[None, :]
        gy = np.linspace(1.0, rng.uniform(0.7, 0.95), H)[:, None]
        if rng.random() < 0.5:
            gx = gx[:, ::-1]
        out *= (gx * gy)[..., None]
    out += np.random.default_rng(rng.randint(0, 10 ** 6)).normal(0, s.noise_sigma, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if abs(skew) > 0.01:
        m = cv2.getRotationMatrix2D((W / 2, H / 2), -skew, 1.0)
        out = cv2.warpAffine(out, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out


def _paste_iam(page: Image.Image, line_img: Image.Image, x: int, y: int, target_h: int, max_w: int):
    line = line_img.convert("L")
    scale = target_h / line.height
    nw = int(line.width * scale)
    if nw > max_w:
        scale = max_w / line.width
        nw = max_w
    line = line.resize((nw, max(1, int(line.height * scale))), Image.LANCZOS)
    arr = np.array(line).astype(np.float32) / 255.0
    # multiply blend: keeps paper/ruling under ink, drops the scan's grey background
    arr = np.clip((arr - 0.15) / 0.75, 0, 1)
    region = np.array(page.crop((x, y, x + line.width, y + line.height))).astype(np.float32)
    blended = region * arr[..., None]
    page.paste(Image.fromarray(blended.astype(np.uint8)), (x, y))
    ys, xs = np.where(arr < 0.6)
    if len(xs) == 0:
        return None
    return [int(x + xs.min()), int(y + ys.min()), int(x + xs.max()), int(y + ys.max())]


def _place_diagram(page: Image.Image, sheet_id: str, cfg: Config, rng: random.Random, out_dir: Path,
                   font_path: str, ink, top: int, margin: int) -> dict:
    """Draw a defective student diagram on the page; save the clean reference; return its GT."""
    W, H = page.size
    ref = dsynth.reference_spec(rng)
    stu = dsynth.student_variant(ref, rng, n_defects=rng.choice([0, 0, 1, 1, 2, 3]))
    dw, dh = int(W * 0.7), int(H * 0.22)
    top = min(top, H - margin - dh)
    dimg = dsynth.render(stu, font_path, (dw, dh), rng, ink=ink)
    dx = (W - dw) // 2
    region = np.array(page.crop((dx, top, dx + dw, top + dh))).astype(np.float32)
    page.paste(Image.fromarray((region * (dimg.astype(np.float32) / 255.0)).astype(np.uint8)), (dx, top))
    ys, xs = np.where(dimg.min(-1) < 160)
    dbox = [int(dx + xs.min()), int(top + ys.min()), int(dx + xs.max()), int(top + ys.max())]
    ref_img = dsynth.render(ref, str(resolve(cfg, cfg.paths.fonts) / "PatrickHand.ttf"), (dw, dh), random.Random(0))
    ref_path = out_dir / f"{sheet_id}_ref_diagram.png"
    Image.fromarray(ref_img).save(ref_path)
    return {"bbox": dbox, "ref_path": str(ref_path), "mark": stu.mark, "defects": stu.defects,
            "ref_spec": asdict(ref), "student_spec": asdict(stu)}


def make_sheet(kind: str, idx: int, cfg: Config, rng: random.Random, out_dir: Path, fonts: list[str],
               iam_pool: list | None = None, asag_row: dict | None = None,
               force: dict | None = None) -> SheetGT:
    s = cfg.synthetic
    force = force or {}
    W, H = s.page_size
    margin = int(s.margin_frac * W)
    spacing = int(H * 0.034)
    ruled = force.get("ruled", rng.random() < s.ruled_prob)
    page = _paper(W, H, rng, ruled, spacing, margin)
    ink = rng.choice(INKS)
    font_path = force.get("font", rng.choice(fonts))
    fsize = rng.randint(*s.font_size_range)
    gt = SheetGT(sheet_id=f"{kind}_{idx:04d}", kind=kind, student_id=f"student_{idx % 23:03d}", image_path="")
    # header the pipeline must not OCR as answer content is kept small and above the first rule
    d = ImageDraw.Draw(page)
    d.text((margin, int(margin * 0.5)), f"Q{(idx % 5) + 1}.", font=ImageFont.truetype(font_path, fsize), fill=ink)
    hdr_box = d.textbbox((margin, int(margin * 0.5)), f"Q{(idx % 5) + 1}.", font=ImageFont.truetype(font_path, fsize))
    gt.lines.append({"bbox": list(hdr_box), "text": f"Q{(idx % 5) + 1}.", "header": True})
    y = margin + spacing * 2
    if kind == "iam":
        n = rng.randint(*s.iam_lines_per_sheet)
        for _ in range(n):
            im, txt = iam_pool[rng.randrange(len(iam_pool))]
            th = int(spacing * 0.9)
            bb = _paste_iam(page, im, margin + rng.randint(0, 30), y - th + int(spacing * 0.2), th, W - 2 * margin)
            if bb:
                gt.lines.append({"bbox": bb, "text": txt})
            y += spacing * rng.choice([1, 1, 2])
    else:
        gt.question, gt.reference = asag_row["question"], asag_row["reference"]
        gt.answer_text, gt.teacher_text_score = asag_row["answer"], float(asag_row["score"])
        lines, y = _wrap_and_render(page, gt.answer_text, font_path, fsize, rng, margin, y, W - 2 * margin,
                                    spacing, ink)
        gt.lines += lines
    if force.get("diagram", kind == "answer" and rng.random() < s.diagram_prob):
        gt.diagram = _place_diagram(page, gt.sheet_id, cfg, rng, out_dir, font_path, ink, y + spacing, margin)
    skew = force.get("skew", rng.uniform(-s.skew_max_deg, s.skew_max_deg) if rng.random() < 0.7 else 0.0)
    shadow = force.get("shadow", rng.random() < s.shadow_prob)
    arr = _degrade(np.array(page), rng, cfg, skew, shadow)
    gt.degradations = {"ruled": ruled, "skew_deg": round(skew, 2), "shadow": shadow,
                       "font": Path(font_path).stem, "font_size": fsize}
    path = out_dir / f"{gt.sheet_id}.png"
    Image.fromarray(arr).save(path)
    gt.image_path = str(path)
    return gt


def font_paths(cfg: Config) -> list[str]:
    d = resolve(cfg, cfg.paths.fonts)
    return [str(d / f"{f}.ttf") for f in cfg.synthetic.fonts if (d / f"{f}.ttf").exists()]


def save_manifest(sheets: list[SheetGT], path: Path) -> None:
    path.write_text(json.dumps([asdict(s) for s in sheets], indent=1))


def load_manifest(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text())

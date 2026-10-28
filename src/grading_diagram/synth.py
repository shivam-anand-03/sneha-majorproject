"""Procedural hand-drawn-style diagrams with controlled defects and rule-based marks.

There is no public dataset of handwritten diagrams with teacher marks, so diagram grading
is evaluated on these synthetic pairs. Marks are deterministic functions of the defects,
which lets us check whether a grader is sensitive to *meaning* (labels, arrows, counts),
not just outline similarity.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TOPICS = {
    "compiler": ["Lexer", "Parser", "Semantic", "Optimizer", "Codegen"],
    "photosynthesis": ["Light", "Chlorophyll", "Glucose", "Oxygen"],
    "network": ["Client", "Router", "Server", "Database"],
    "water_cycle": ["Evaporation", "Condensation", "Rain", "Runoff"],
    "cpu": ["Fetch", "Decode", "Execute", "Store"],
    "tree": ["Root", "Left", "Right", "Leaf"],
}
LAYOUTS = ["chain_h", "chain_v", "tree"]

# penalty per defect (fraction of full diagram marks)
PENALTY = {"drop_component": 0.30, "drop_label": 0.15, "wrong_label": 0.15, "drop_arrow": 0.12,
           "reverse_arrow": 0.10, "shuffle": 0.25}


@dataclass
class DiagramSpec:
    topic: str
    layout: str
    nodes: list[str | None]          # label per node; None = unlabelled
    edges: list[tuple[int, int]]     # directed
    defects: list[str] = field(default_factory=list)

    @property
    def mark(self) -> float:
        return float(max(0.0, 1.0 - sum(PENALTY[d] for d in self.defects)))


def reference_spec(rng: random.Random) -> DiagramSpec:
    topic = rng.choice(list(TOPICS))
    layout = rng.choice(LAYOUTS)
    nodes = list(TOPICS[topic])
    if layout == "tree":
        edges = [(0, i) for i in range(1, len(nodes))]
    else:
        edges = [(i, i + 1) for i in range(len(nodes) - 1)]
    return DiagramSpec(topic, layout, nodes, edges)


def student_variant(ref: DiagramSpec, rng: random.Random, n_defects: int) -> DiagramSpec:
    s = DiagramSpec(ref.topic, ref.layout, list(ref.nodes), list(ref.edges))
    options = list(PENALTY)
    for _ in range(n_defects):
        d = rng.choice(options)
        if d == "drop_component" and len(s.nodes) > 2:
            k = rng.randrange(1, len(s.nodes))
            s.nodes.pop(k)
            s.edges = [(a - (a > k), b - (b > k)) for a, b in s.edges if k not in (a, b)]
            if s.layout != "tree":
                s.edges = [(i, i + 1) for i in range(len(s.nodes) - 1)]
        elif d == "drop_label":
            idx = [i for i, n in enumerate(s.nodes) if n]
            if not idx:
                continue
            s.nodes[rng.choice(idx)] = None
        elif d == "wrong_label":
            idx = [i for i, n in enumerate(s.nodes) if n]
            pool = [w for t, ws in TOPICS.items() if t != s.topic for w in ws]
            if not idx:
                continue
            s.nodes[rng.choice(idx)] = rng.choice(pool)
        elif d == "drop_arrow" and s.edges:
            s.edges.pop(rng.randrange(len(s.edges)))
        elif d == "reverse_arrow" and s.edges:
            i = rng.randrange(len(s.edges))
            a, b = s.edges[i]
            s.edges[i] = (b, a)
        elif d == "shuffle" and len(s.nodes) > 2:
            labs = [n for n in s.nodes]
            rng.shuffle(labs)
            if labs == s.nodes:
                continue
            s.nodes = labs
        else:
            continue
        s.defects.append(d)
    return s


def _wobbly_line(draw: ImageDraw.ImageDraw, p0, p1, rng: random.Random, width: int, fill):
    n = 8
    pts = []
    for t in np.linspace(0, 1, n):
        x = p0[0] + (p1[0] - p0[0]) * t + rng.uniform(-2, 2)
        y = p0[1] + (p1[1] - p0[1]) * t + rng.uniform(-2, 2)
        pts.append((x, y))
    draw.line(pts, fill=fill, width=width, joint="curve")


def _node_positions(spec: DiagramSpec, W: int, H: int) -> list[tuple[int, int]]:
    n = len(spec.nodes)
    if spec.layout == "chain_h":
        return [(int(W * (i + 0.5) / n), H // 2) for i in range(n)]
    if spec.layout == "chain_v":
        return [(W // 2, int(H * (i + 0.5) / n)) for i in range(n)]
    pos = [(W // 2, int(H * 0.2))]
    kids = n - 1
    pos += [(int(W * (i + 0.5) / max(kids, 1)), int(H * 0.75)) for i in range(kids)]
    return pos


def render(spec: DiagramSpec, font_path: str, size: tuple[int, int], rng: random.Random,
           ink=(20, 30, 90), line_width: int = 4) -> np.ndarray:
    """Hand-drawn style rendering on a transparent-white canvas (RGB uint8)."""
    W, H = size
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    n = len(spec.nodes)
    pos = _node_positions(spec, W, H)
    if spec.layout == "chain_h":
        bw, bh = int(W / n * 0.7), int(H * 0.35)
    elif spec.layout == "chain_v":
        bw, bh = int(W * 0.45), int(H / n * 0.55)
    else:
        bw, bh = int(W / max(n - 1, 1) * 0.7), int(H * 0.25)
    fsize = max(18, int(min(bh * 0.45, bw / 7)))
    font = ImageFont.truetype(font_path, fsize)
    boxes = []
    for (cx, cy), label in zip(pos, spec.nodes):
        jx, jy = rng.uniform(-0.05, 0.05) * bw, rng.uniform(-0.05, 0.05) * bh
        x0, y0, x1, y1 = cx - bw / 2 + jx, cy - bh / 2 + jy, cx + bw / 2 + jx, cy + bh / 2 + jy
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        for a, b in zip(corners, corners[1:]):
            _wobbly_line(d, a, b, rng, line_width, ink)
        boxes.append((x0, y0, x1, y1))
        if label:
            tw = d.textlength(label, font=font)
            d.text((cx + jx - tw / 2, cy + jy - fsize * 0.6), label, font=font, fill=ink)
    for a, b in spec.edges:
        if a >= len(boxes) or b >= len(boxes):
            continue
        ax0, ay0, ax1, ay1 = boxes[a]
        bx0, by0, bx1, by1 = boxes[b]
        acx, acy = (ax0 + ax1) / 2, (ay0 + ay1) / 2
        bcx, bcy = (bx0 + bx1) / 2, (by0 + by1) / 2
        dx, dy = bcx - acx, bcy - acy
        # leave from box edge, arrive at box edge
        if abs(dx) * bh > abs(dy) * bw:
            p0 = (ax1 if dx > 0 else ax0, acy)
            p1 = (bx0 if dx > 0 else bx1, bcy)
        else:
            p0 = (acx, ay1 if dy > 0 else ay0)
            p1 = (bcx, by0 if dy > 0 else by1)
        _wobbly_line(d, p0, p1, rng, line_width, ink)
        ang = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
        L = max(12, line_width * 4)
        for s in (+0.45, -0.45):
            q = (p1[0] - L * np.cos(ang + s), p1[1] - L * np.sin(ang + s))
            d.line([p1, q], fill=ink, width=line_width)
    arr = np.array(img)
    return cv2.GaussianBlur(arr, (3, 3), 0)

"""Generate synthetic answer sheets with ground truth.

python -m scripts.make_synthetic --n-iam 30 --n-answer 60
"""
from __future__ import annotations

import argparse
import random

from src.common.config import load_config, resolve, seed_everything
from src.data.asag import load_mohler
from src.data.iam import load_iam_lines
from src.data.synthetic import font_paths, make_sheet, save_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iam", type=int, default=30)
    ap.add_argument("--n-answer", type=int, default=60)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.project.seed)
    rng = random.Random(cfg.project.seed)
    out = resolve(cfg, cfg.paths.synthetic)
    out.mkdir(parents=True, exist_ok=True)
    fonts = font_paths(cfg)
    sheets = []
    if args.n_iam:
        # IAM *test* split only: writer-disjoint from IAM train used by TrOCR
        pool = load_iam_lines("test", limit=600, seed=cfg.project.seed)
        sheets += [make_sheet("iam", i, cfg, rng, out, fonts, iam_pool=pool) for i in range(args.n_iam)]
    if args.n_answer:
        moh = load_mohler(cfg)
        test = moh[moh.split == "test"].sample(n=min(args.n_answer, (moh.split == "test").sum()),
                                               random_state=cfg.project.seed)
        sheets += [make_sheet("answer", i, cfg, rng, out, fonts, asag_row=row.to_dict())
                   for i, (_, row) in enumerate(test.iterrows())]
    save_manifest(sheets, out / "manifest.json")
    n_diag = sum(1 for s in sheets if s.diagram)
    print(f"wrote {len(sheets)} sheets ({n_diag} with diagrams) to {out}")


if __name__ == "__main__":
    main()

"""RQ2 for several systems at once: does dialect distance (100 - copy chrF) predict difficulty?

Per sentence: Spearman rho between the copy-based distance and the system's sentence chrF (runs averaged),
with a 95% bootstrap CI. Per dialect: Spearman rho between the five dialect distances and the five
dialect-level corpus chrF scores.

  python src/rq2_all_systems.py --split test --systems dictionary banglat5_cpu1ep qwen05b_lora \
      --write results/test/rq2_all_systems.md
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from common import DIALECTS, load_jsonl
from evaluate import corpus_chrf, sentence_chrf, spearman_ci


def load(out_dir, system, split):
    return [load_jsonl(p) for p in sorted(Path(out_dir, system).glob(f"*/{split}.jsonl"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--write", default=None)
    args = ap.parse_args()
    copy = load(args.out_dir, "copy", args.split)[0]
    dist = dict(zip([r["id"] for r in copy], [100 - s for s in sentence_chrf([r["hyp"] for r in copy],
                                                                          [r["tgt"] for r in copy])]))
    by_d = defaultdict(list)
    for r in copy:
        by_d[r["dialect"]].append(r)
    dialects = [d for d in DIALECTS if d in by_d]
    ddist = [100 - corpus_chrf([r["hyp"] for r in by_d[d]], [r["tgt"] for r in by_d[d]]) for d in dialects]
    lines = [f"# RQ2 on {args.split}: distance vs. difficulty for every system\n",
             "Dialect distance: " + ", ".join(f"{d} {x:.1f}" for d, x in zip(dialects, ddist)) + "\n",
             "| System | Sentence-level ρ | 95% CI | n | Dialect-level ρ (5 dialects) | Hardest → easiest |",
             "|---|---|---|---|---|---|"]
    for system in args.systems:
        runs = load(args.out_dir, system, args.split)
        if not runs:
            continue
        per_run = [dict(zip([r["id"] for r in run], sentence_chrf([r["hyp"] for r in run], [r["tgt"] for r in run])))
                   for run in runs]
        ids = sorted(set(dist).intersection(*per_run))
        x = [dist[i] for i in ids]
        y = [np.mean([pr[i] for pr in per_run]) for i in ids]
        rho, lo, hi = spearman_ci(x, y)
        dchrf = []
        for d in dialects:
            sc = []
            for run in runs:
                rows = [r for r in run if r["dialect"] == d]
                sc.append(corpus_chrf([r["hyp"] for r in rows], [r["tgt"] for r in rows]))
            dchrf.append(np.mean(sc))
        drho = spearmanr(ddist, dchrf)[0]
        order = ", ".join(d[:4].capitalize() for _, d in sorted(zip(dchrf, dialects)))
        lines.append(f"| {system} | {rho:.3f} | [{lo:.3f}, {hi:.3f}] | {len(ids)} | {drho:.2f} | {order} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.write:
        Path(args.write).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

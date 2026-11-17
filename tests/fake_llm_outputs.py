"""Writes fake LLM outputs (noisy copies of the reference) so evaluate / human_eval / error_analysis
can be tested without a GPU. Never use these numbers in the report."""
import argparse
import random
from pathlib import Path

from common import load_jsonl, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    for split in ("val", "test"):
        rows = load_jsonl(Path(args.data_dir) / f"{split}.jsonl")
        for system, runs, p in (("qwen7b_zeroshot", ["p0", "p1", "p2"], 0.5), ("qwen7b_fewshot", ["s1", "s2", "s3"], 0.3),
                                ("qwen7b_qlora", ["s1", "s2", "s3"], 0.1)):
            for k, run in enumerate(runs):
                rng = random.Random(f"{system}{run}")
                out = []
                for r in rows:
                    words = [w for w in r["tgt"].split() if rng.random() > p / 2]
                    hyp = r["src"] if rng.random() < p / 2 else " ".join(words)
                    if rng.random() < 0.02:
                        hyp = ""  # unusable output
                    out.append({**r, "raw": hyp, "hyp": hyp})
                write_jsonl(Path(args.out_dir) / system / run / f"{split}.jsonl", out)
    print("fake outputs written")


if __name__ == "__main__":
    main()

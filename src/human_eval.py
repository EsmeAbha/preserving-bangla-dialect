"""Blind human meaning ratings (1-5) and agreement.

make   picks 150 test outputs: the same sentences for each compared system, stratified by dialect.
       It writes a shuffled sheet for annotator 1 (all 150), a separately shuffled sheet for
       annotator 2 (50 of them, one third), and a key that is not shown to annotators.
score  reads the filled sheets: mean rating per system (and per dialect), quadratic-weighted Cohen's
       kappa on the 50 shared items, and Spearman correlation of ratings with sentence chrF / LaBSE.

  python src/human_eval.py make --systems qwen7b_zeroshot qwen7b_fewshot qwen7b_qlora --runs p0 s1 s1
  python src/human_eval.py score            (after filling the `rating` column in both sheets)

Rating scale (put it in front of the annotators):
  5 same meaning   4 minor difference   3 main meaning kept, a detail lost or wrong
  2 meaning largely changed   1 unrelated / unusable / still unreadable dialect
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from common import load_jsonl


def quadratic_kappa(a, b, k_min=1, k_max=5):
    a, b = np.asarray(a, int) - k_min, np.asarray(b, int) - k_min
    k = k_max - k_min + 1
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[x, y] += 1
    exp = np.outer(obs.sum(1), obs.sum(0)) / obs.sum()
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    return 1 - (w * obs).sum() / (w * exp).sum()


def make(args):
    if len(args.runs) != len(args.systems):
        raise SystemExit("--runs needs one run name per system")
    data = [{r["id"]: r for r in load_jsonl(Path(args.out_dir) / s / run / f"{args.split}.jsonl")}
            for s, run in zip(args.systems, args.runs)]
    shared = sorted(set.intersection(*[set(d) for d in data]))
    per_sys = args.n // len(args.systems)
    by_dialect = defaultdict(list)
    for i in shared:
        by_dialect[data[0][i]["dialect"]].append(i)
    rng = random.Random(args.seed)
    per_d = per_sys // len(by_dialect)
    chosen = []
    for d in sorted(by_dialect):
        chosen += rng.sample(by_dialect[d], per_d)
    items = []
    for s, run, d in zip(args.systems, args.runs, data):
        for i in chosen:
            r = d[i]
            items.append({"system": s, "run": run, "id": i, "dialect": r["dialect"], "src": r["src"],
                          "tgt": r["tgt"], "hyp": r["hyp"]})
    rng.shuffle(items)
    for k, it in enumerate(items, 1):
        it["item"] = f"H{k:03d}"
    out = Path(args.he_dir)
    out.mkdir(parents=True, exist_ok=True)

    def sheet(path, rows):
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:  # BOM so Excel shows Bangla
            w = csv.writer(fh)
            w.writerow(["item", "dialect", "dialect_sentence", "standard_reference", "system_output", "rating"])
            for it in rows:
                w.writerow([it["item"], it["dialect"], it["src"], it["tgt"], it["hyp"], ""])

    sheet(out / "sheet_annotator1.csv", items)
    second = rng.sample(items, args.n_second)
    rng.shuffle(second)
    sheet(out / "sheet_annotator2.csv", second)
    with open(out / "key.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["item", "system", "run", "id", "dialect"])
        for it in sorted(items, key=lambda x: x["item"]):
            w.writerow([it["item"], it["system"], it["run"], it["id"], it["dialect"]])
    print(f"{len(items)} items ({len(chosen)} sentences x {len(args.systems)} systems); "
          f"{len(second)} for annotator 2 -> {out}/")


def read_sheet(path):
    with open(path, encoding="utf-8-sig") as fh:
        return {r["item"]: int(r["rating"]) for r in csv.DictReader(fh) if r["rating"].strip()}


def score(args):
    out = Path(args.he_dir)
    key = {r["item"]: r for r in csv.DictReader(open(out / "key.csv", encoding="utf-8"))}
    a1, a2 = read_sheet(out / "sheet_annotator1.csv"), read_sheet(out / "sheet_annotator2.csv")
    lines = ["# Human meaning ratings\n", f"Annotator 1: {len(a1)} items; annotator 2: {len(a2)} items.\n"]
    shared = sorted(set(a1) & set(a2))
    if shared:
        x, y = [a1[i] for i in shared], [a2[i] for i in shared]
        lines.append(f"Quadratic-weighted Cohen's κ on {len(shared)} shared items: {quadratic_kappa(x, y):.3f} "
                     f"(exact agreement {np.mean(np.array(x) == np.array(y)):.2f})\n")
    by_sys = defaultdict(list)
    by_sys_d = defaultdict(list)
    for i, r in a1.items():
        by_sys[key[i]["system"]].append(r)
        by_sys_d[(key[i]["system"], key[i]["dialect"])].append(r)
    dialects = sorted({k["dialect"] for k in key.values()})
    lines += ["| System | n | Mean rating | " + " | ".join(dialects) + " |", "|---" * (3 + len(dialects)) + "|"]
    for s, v in sorted(by_sys.items()):
        lines.append(f"| {s} | {len(v)} | {np.mean(v):.2f} ± {np.std(v, ddof=1):.2f} | "
                     + " | ".join(f"{np.mean(by_sys_d[(s, d)]):.2f}" for d in dialects) + " |")
    sent = Path(args.results_dir) / args.split / "sentences.csv"
    if sent.exists():
        auto = {(r["system"], r["run"], r["id"]): r for r in csv.DictReader(open(sent, encoding="utf-8"))}
        pairs = [(a1[i], auto.get((key[i]["system"], key[i]["run"], key[i]["id"]))) for i in a1]
        pairs = [(h, a) for h, a in pairs if a]
        for m in ("chrF", "LaBSE"):
            vals = [(h, float(a[m])) for h, a in pairs if a[m] not in ("", None)]
            if len(vals) > 2:
                rho = spearmanr(*zip(*vals)).statistic
                lines.append(f"\nSpearman ρ between human rating and sentence {m}: {rho:.3f} (n={len(vals)})")
    (out / "human_eval_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--systems", nargs="+", default=["qwen7b_zeroshot", "qwen7b_fewshot", "qwen7b_qlora"])
    m.add_argument("--runs", nargs="+", default=["p0", "s1", "s1"])
    m.add_argument("--n", type=int, default=150)
    m.add_argument("--n_second", type=int, default=50)
    m.add_argument("--seed", type=int, default=7)
    s = sub.add_parser("score")
    for p in (m, s):
        p.add_argument("--out_dir", default="outputs")
        p.add_argument("--results_dir", default="results")
        p.add_argument("--he_dir", default="human_eval")
        p.add_argument("--split", default="test")
    args = ap.parse_args()
    make(args) if args.cmd == "make" else score(args)


if __name__ == "__main__":
    main()

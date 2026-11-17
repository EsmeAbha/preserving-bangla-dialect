"""Error analysis of the main model.

sample     draws a random sample of wrong outputs (output differs from the reference) from one run,
           stratified by dialect, into a CSV with an empty `category` column to fill in by hand.
summarize  counts the categories per dialect and lists one example per category.

Categories: word_sense (wrong word sense), dropped (dropped words), negation_tense (changed negation
or tense), still_dialect (output still in dialect), other.

  python error_analysis.py sample --system qwen7b_qlora --run s1 --n 60
  python error_analysis.py summarize
"""
import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from sacrebleu.metrics import CHRF

from common import load_jsonl

CATEGORIES = ["word_sense", "dropped", "negation_tense", "still_dialect", "other"]


def norm(s):
    return " ".join(s.replace("।", " ").replace("?", " ").replace(",", " ").split())


def sample(args):
    rows = load_jsonl(Path(args.out_dir) / args.system / args.run / f"{args.split}.jsonl")
    wrong = [r for r in rows if norm(r["hyp"] or "") != norm(r["tgt"])]
    by_d = defaultdict(list)
    for r in wrong:
        by_d[r["dialect"]].append(r)
    rng = random.Random(args.seed)
    per = args.n // len(by_d)
    chosen = [r for d in sorted(by_d) for r in rng.sample(by_d[d], min(per, len(by_d[d])))]
    chrf = CHRF()
    path = Path(args.ea_dir) / f"errors_{args.system}_{args.run}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "dialect", "dialect_sentence", "reference", "output", "chrF", "category", "note"])
        for r in chosen:
            w.writerow([r["id"], r["dialect"], r["src"], r["tgt"], r["hyp"],
                        round(chrf.sentence_score(r["hyp"] or "", [r["tgt"]]).score, 1), "", ""])
    print(f"{len(wrong)}/{len(rows)} outputs differ from the reference; sampled {len(chosen)} -> {path}")
    print("fill `category` with one of:", ", ".join(CATEGORIES))


def summarize(args):
    lines = []
    for path in sorted(Path(args.ea_dir).glob("errors_*.csv")):
        rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
        done = [r for r in rows if r["category"].strip()]
        bad = {r["category"] for r in done} - set(CATEGORIES)
        if bad:
            print(f"{path.name}: unknown categories {bad}")
        dialects = sorted({r["dialect"] for r in done})
        cnt = Counter((r["category"].strip(), r["dialect"]) for r in done)
        tot = Counter(r["category"].strip() for r in done)
        lines += [f"## {path.stem} ({len(done)} labelled errors)\n",
                  "| Category | " + " | ".join(dialects) + " | Total | % |", "|---" * (len(dialects) + 3) + "|"]
        for c in CATEGORIES + sorted(bad):
            if tot[c]:
                lines.append(f"| {c} | " + " | ".join(str(cnt[(c, d)]) for d in dialects)
                             + f" | {tot[c]} | {100 * tot[c] / len(done):.0f} |")
        lines.append("\nExamples:\n")
        for c in CATEGORIES:
            ex = next((r for r in done if r["category"].strip() == c), None)
            if ex:
                lines.append(f"- **{c}** ({ex['dialect']}): {ex['dialect_sentence']} → {ex['output']} "
                             f"(ref: {ex['reference']}) {ex['note']}")
        lines.append("")
    out = Path(args.ea_dir) / "error_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--system", default="qwen7b_qlora")
    s.add_argument("--run", default="s1")
    s.add_argument("--n", type=int, default=60)
    s.add_argument("--seed", type=int, default=3)
    s.add_argument("--split", default="test")
    s.add_argument("--out_dir", default="outputs")
    m = sub.add_parser("summarize")
    for p in (s, m):
        p.add_argument("--ea_dir", default="error_analysis")
    args = ap.parse_args()
    sample(args) if args.cmd == "sample" else summarize(args)


if __name__ == "__main__":
    main()

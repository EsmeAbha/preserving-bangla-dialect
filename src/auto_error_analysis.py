"""Automatic word-level error profile of system outputs (complements the manual error analysis).

For each sentence, the *dialect words* are the source words that do not occur in the reference, i.e. the
words that had to be changed. Reported per system:
  exact        output identical to the reference (ignoring spaces and punctuation)
  kept         share of dialect words left unchanged in the output (still-dialect errors)
  fixed        share of the reference words missing from the source (the words that had to be produced)
               that appear in the output
  novel        share of output words found in neither the source nor the reference (invented or
               paraphrased words)
  neg          share of sentences where the reference and the output disagree on negation
               (negators না, নি, নাই, নেই, নয়)
  len          output length / reference length in words

  python src/auto_error_analysis.py --split val --systems copy dictionary qwen3b_fewshot ...
"""
import argparse
import re
from pathlib import Path

from common import load_jsonl

WORD = re.compile(r"[ঀ-৿A-Za-z0-9]+")
NEG = {"না", "নি", "নাই", "নেই", "নয়", "নয়"}  # নয় in two Unicode spellings


def words(text):
    return WORD.findall(text)


def profile(rows):
    exact = kept_n = kept_d = fix_n = fix_d = nov_n = nov_d = neg = 0
    lr = []
    for r in rows:
        s, t, h = words(r["src"]), words(r["tgt"]), words(r.get("hyp", ""))
        S, T, H = set(s), set(t), set(h)
        exact += "".join(h) == "".join(t)
        dia = [w for w in s if w not in T]
        kept_n += sum(w in H for w in dia)
        kept_d += len(dia)
        need = [w for w in t if w not in S]
        fix_n += sum(w in H for w in need)
        fix_d += len(need)
        nov_n += sum(w not in S and w not in T for w in h)
        nov_d += len(h)
        neg += bool(NEG & T) != bool(NEG & H)
        lr.append(len(h) / max(1, len(t)))
    n = len(rows)
    pct = lambda a, b: 100 * a / max(1, b)
    return {"exact": pct(exact, n), "kept": pct(kept_n, kept_d), "fixed": pct(fix_n, fix_d),
            "novel": pct(nov_n, nov_d), "neg": pct(neg, n), "len": sum(lr) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--write", default=None, help="markdown file to write")
    args = ap.parse_args()
    lines = [f"# Automatic error profile on {args.split} (mean over runs)\n",
             "| System | Exact % | Dialect words kept % | Needed words produced % | Novel words % | "
             "Negation mismatch % | Length ratio |", "|---|---|---|---|---|---|---|"]
    for sysname in args.systems:
        runs = sorted(Path(args.out_dir, sysname).glob(f"*/{args.split}.jsonl"))
        if not runs:
            print("no outputs for", sysname)
            continue
        ps = [profile(load_jsonl(p)) for p in runs]
        m = {k: sum(p[k] for p in ps) / len(ps) for k in ps[0]}
        lines.append(f"| {sysname} | {m['exact']:.1f} | {m['kept']:.1f} | {m['fixed']:.1f} | {m['novel']:.1f} | "
                     f"{m['neg']:.1f} | {m['len']:.2f} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.write:
        Path(args.write).parent.mkdir(parents=True, exist_ok=True)
        Path(args.write).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

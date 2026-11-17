"""Prepare ONUBAD (Mendeley Data V2, doi:10.17632/6ft99kf89b.2, CC BY 4.0) as an out-of-domain test set.

Only Chittagong, Sylhet and Barishal are in ONUBAD. Word- and clause-level entries are not used:
if the files have a column describing the entry type (e.g. "type"/"level"), only sentence rows are
kept; otherwise rows with fewer than --min_words words in the standard sentence are dropped.

Column detection (check the printed choices):
  * a column whose name contains a dialect name (chittagong/chattogram/ctg, sylhet, barishal/barisal)
    is that dialect's text;
  * the standard column is the one whose name contains "standard", "shuddho", "promito" or "bangla"/"bengali"
    and no dialect name and not "english". Override with --std_col and --dialect_cols.
  * if a file has no dialect-named column, the dialect is taken from the file name and --src_col is used
    (or the single remaining non-English text column).

  python src/prepare_onubad.py --raw_dir data/raw_onubad --out data/processed/onubad.jsonl
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from prepare_data import group_key, normalise

ALIASES = {"chittagong": ["chittagong", "chattogram", "ctg", "chatgaiya"],
           "sylhet": ["sylhet", "syloti", "sylheti"],
           "barishal": ["barishal", "barisal"]}
STD_WORDS = ["standard", "shuddho", "promito", "bangla", "bengali"]


def which_dialect(name):
    n = name.lower()
    for d, al in ALIASES.items():
        if any(a in n for a in al):
            return d
    return None


def read_any(f):
    if f.suffix.lower() in (".xlsx", ".xls"):
        return pd.concat(pd.read_excel(f, sheet_name=None).values(), ignore_index=True)
    return pd.read_csv(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out", default="data/processed/onubad.jsonl")
    ap.add_argument("--std_col", default=None)
    ap.add_argument("--src_col", default=None, help="dialect column when the dialect is given by the file name")
    ap.add_argument("--dialect_cols", nargs="*", default=[], help="e.g. chittagong=Chittagong_Text sylhet=...")
    ap.add_argument("--type_col", default=None, help="column with the entry level (word/clause/sentence)")
    ap.add_argument("--min_words", type=int, default=3)
    ap.add_argument("--vashantor_train", default="data/processed/train.jsonl")
    args = ap.parse_args()
    manual = dict(x.split("=", 1) for x in args.dialect_cols)

    rows, dropped = [], Counter()
    files = [f for f in Path(args.raw_dir).rglob("*") if f.suffix.lower() in (".csv", ".xlsx", ".xls")]
    for f in sorted(files):
        df = read_any(f)
        cols = [str(c) for c in df.columns]
        df.columns = cols
        std = args.std_col or next((c for c in cols if any(w in c.lower() for w in STD_WORDS)
                                    and not which_dialect(c) and "english" not in c.lower()), None)
        dcols = {d: c for d, c in manual.items() if c in cols} or {which_dialect(c): c for c in cols if which_dialect(c)}
        if not dcols:
            d = which_dialect(f.name)
            other = [c for c in cols if c != std and "english" not in c.lower()
                     and (df[c].dtype == object or pd.api.types.is_string_dtype(df[c]))]
            src = args.src_col or (other[0] if len(other) == 1 else None)
            if d and src:
                dcols = {d: src}
        if not std or not dcols:
            print(f"skip {f.name}: columns {cols} (pass --std_col / --dialect_cols / --src_col)")
            continue
        tcol = args.type_col or next((c for c in cols if c.lower() in ("type", "level", "category", "unit")), None)
        print(f"{f.name}: standard={std!r} dialects={dcols} type={tcol!r} rows={len(df)}")
        for _, r in df.iterrows():
            if tcol and "sentence" not in str(r[tcol]).lower():
                dropped["not_sentence_type"] += 1
                continue
            t = normalise(r[std])
            if len(t.split()) < args.min_words:
                dropped["short_or_empty"] += 1
                continue
            for d, c in dcols.items():
                s = normalise(r[c])
                if s:
                    rows.append({"dialect": d, "src": s, "tgt": t})

    seen, uniq = set(), []
    for r in rows:
        k = (r["dialect"], r["src"], r["tgt"])
        if k in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(k)
        uniq.append(r)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for i, r in enumerate(uniq):
            fh.write(json.dumps({"id": f"onubad-{i:05d}", "group_id": -1, **r}, ensure_ascii=False) + "\n")
    stats = {"pairs": len(uniq), "per_dialect": dict(Counter(r["dialect"] for r in uniq)), "dropped": dict(dropped)}
    vt = Path(args.vashantor_train)
    if vt.exists():
        train_keys = {group_key(json.loads(l)["tgt"]) for l in open(vt, encoding="utf-8")}
        stats["standard_sentences_also_in_vashantor_train"] = sum(group_key(r["tgt"]) in train_keys for r in uniq)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    json.dump(stats, open(out.with_suffix(".stats.json"), "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

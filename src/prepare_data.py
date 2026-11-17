"""Prepare Vashantor (and optionally ONUBAD) for CSC5515 G14.

Vashantor: Mendeley Data V2, doi:10.17632/bj5jgk878b.2, CC BY 4.0. Download and unzip
Vashantor_CSV_Format.zip. Only the Bangla-script dialect files are used (not Banglish or English).

What the script does
  1. Finds, for each dialect (Chittagong, Noakhali, Sylhet, Barishal, Mymensingh) and each split
     (train, validation, test), the CSV with the dialect sentence and its standard Bangla sentence.
     File names must contain the dialect name and a split word (train / val / test); files whose
     name contains "banglish" or "english" are skipped. Column names are detected automatically
     and can be overridden with --src_col and --tgt_col.
  2. Cleans text: Unicode NFC, zero-width characters removed, whitespace collapsed. Empty pairs and
     exact duplicate pairs within a dialect and split are removed and counted.
  3. Aligns the splits by standard sentence. All pairs that share the same normalised standard
     sentence (ignoring punctuation and spaces) form one group, and each group must sit in a single split for all dialects, so that a
     held-out target is never seen in training through another dialect (needed for the
     leave-one-dialect-out test). A group found in more than one split is moved entirely to its
     test split if any member is in test, otherwise to validation. Moved pairs are counted.
  4. Writes data/processed/{train,val,test}.jsonl and data/processed/stats.json, and prints the
     copy-baseline chrF per dialect on validation (the word-level dialect distance is 100 minus it).

Usage:
  python src/prepare_data.py --raw_dir <unzipped Vashantor CSV folder> --out_dir data/processed
"""
import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

DIALECTS = ["chittagong", "noakhali", "sylhet", "barishal", "mymensingh"]
SPLITS = {"train": ["train"], "val": ["val"], "test": ["test"]}
SKIP_WORDS = ["banglish", "english"]
ZERO_WIDTH = re.compile("[​‌‍⁠﻿]")


def normalise(text) -> str:
    if pd.isna(text):
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = ZERO_WIDTH.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


PUNCT_KEY = re.compile(r"[\s।?!,;:.\"'“”‘’()\-–—]+")


def group_key(text: str) -> str:
    """Key for aligning splits: the normalised standard sentence without punctuation and spaces,
    so that e.g. "... নেই" and "... নেই।" count as the same standard sentence."""
    return PUNCT_KEY.sub("", text)


def find_file(raw: Path, dialect: str, split: str) -> Path:
    hits = []
    for f in raw.rglob("*.csv"):
        name = str(f.relative_to(raw)).lower()
        if dialect in name and any(w in name for w in SPLITS[split]) and not any(w in name for w in SKIP_WORDS):
            hits.append(f)
    if len(hits) != 1:
        raise SystemExit(f"Expected one file for {dialect}/{split}, found {len(hits)}: {hits}. "
                         "Rename the files or move the Banglish/English files out of --raw_dir.")
    return hits[0]


def pick_columns(df: pd.DataFrame, dialect: str, src_col, tgt_col):
    if src_col and tgt_col:
        return src_col, tgt_col
    text_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c])]
    if len(text_cols) != 2:
        raise SystemExit(f"{dialect}: found text columns {text_cols}; pass --src_col and --tgt_col.")
    a, b = text_cols
    # the dialect column usually carries the dialect name; otherwise assume (dialect, standard)
    if dialect in b.lower() and dialect not in a.lower():
        a, b = b, a
    return a, b


def chrf(hyps, refs):
    try:
        from sacrebleu.metrics import CHRF
    except ImportError:
        return None
    return round(CHRF().corpus_score(hyps, [refs]).score, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--src_col", default=None, help="column with the dialect sentence")
    ap.add_argument("--tgt_col", default=None, help="column with the standard Bangla sentence")
    args = ap.parse_args()
    raw, out = Path(args.raw_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows, removed = [], Counter()
    for dialect in DIALECTS:
        for split in SPLITS:
            f = find_file(raw, dialect, split)
            df = pd.read_csv(f)
            sc, tc = pick_columns(df, dialect, args.src_col, args.tgt_col)
            print(f"{dialect:>10} {split:<5} {len(df):>5} rows  file={f.name}  src={sc!r}  tgt={tc!r}")
            seen = set()
            for s, t in zip(df[sc].map(normalise), df[tc].map(normalise)):
                if not s or not t:
                    removed["empty"] += 1
                    continue
                if (s, t) in seen:
                    removed["duplicate"] += 1
                    continue
                seen.add((s, t))
                rows.append({"dialect": dialect, "split": split, "src": s, "tgt": t})

    # align splits by standard sentence
    splits_of = defaultdict(set)
    for r in rows:
        r["key"] = group_key(r["tgt"])
        splits_of[r["key"]].add(r["split"])
    moved = 0
    for r in rows:
        sp = splits_of[r["key"]]
        if len(sp) > 1:
            new = "test" if "test" in sp else "val"
            if r["split"] != new:
                r["split"] = new
                moved += 1
    groups = {t: i for i, t in enumerate(sorted(splits_of))}

    for name in SPLITS:
        part = [r for r in rows if r["split"] == name]
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for i, r in enumerate(part):
                rec = {"id": f"{name}-{i:05d}", "group_id": groups[r["key"]], **{k: r[k] for k in ("dialect", "src", "tgt")}}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # checks: no standard sentence crosses splits
    by_split = {n: {r["key"] for r in rows if r["split"] == n} for n in SPLITS}
    assert not (by_split["test"] & by_split["train"]) and not (by_split["val"] & by_split["train"]) \
        and not (by_split["test"] & by_split["val"]), "split alignment failed"

    stats = {"removed": dict(removed), "moved_for_alignment": moved, "standard_sentence_groups": len(groups),
             "group_size_counts": dict(Counter(Counter(r["key"] for r in rows).values())), "splits": {}}
    for name in SPLITS:
        part = [r for r in rows if r["split"] == name]
        stats["splits"][name] = {
            "pairs": len(part),
            "per_dialect": dict(Counter(r["dialect"] for r in part)),
            "standard_sentences": len(by_split[name]),
            "mean_src_words": round(sum(len(r["src"].split()) for r in part) / max(len(part), 1), 1),
            "mean_tgt_words": round(sum(len(r["tgt"].split()) for r in part) / max(len(part), 1), 1),
        }
    val = [r for r in rows if r["split"] == "val"]
    stats["copy_baseline_chrf_val"] = {d: chrf([r["src"] for r in val if r["dialect"] == d],
                                               [r["tgt"] for r in val if r["dialect"] == d]) for d in DIALECTS}
    with open(out / "stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Non-LLM baselines: copy and word-level dictionary.

copy        the dialect input is output unchanged (also gives the dialect distance: 100 - chrF).
dictionary  word replacements learned per dialect from aligned training pairs (difflib alignment of
            the two token sequences); the most frequent standard form is used, unknown words are copied.

Usage:
  python src/baselines.py --data_dir data/processed --out_dir outputs --splits val test
Writes outputs/{copy,dictionary}/run1/{split}.jsonl
"""
import argparse
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from common import load_jsonl, write_jsonl

PUNCT = re.compile(r"([।?!,;:\"'“”‘’()\-–—.])")


def tokenize(text):
    return PUNCT.sub(r" \1 ", text).split()


def detokenize(tokens):
    text = " ".join(tokens)
    text = re.sub(r"\s+([।?!,;:.)”’])", r"\1", text)
    return re.sub(r"([(“‘])\s+", r"\1", text).strip()


class Dictionary:
    def __init__(self):
        self.table = defaultdict(lambda: defaultdict(Counter))  # dialect -> src word -> Counter(tgt)

    def fit(self, rows):
        for r in rows:
            s, t = tokenize(r["src"]), tokenize(r["tgt"])
            for op, i1, i2, j1, j2 in SequenceMatcher(None, s, t, autojunk=False).get_opcodes():
                if op == "equal" or (op == "replace" and i2 - i1 == j2 - j1):
                    for a, b in zip(s[i1:i2], t[j1:j2]):
                        self.table[r["dialect"]][a][b] += 1
                elif op == "replace" and i2 - i1 == 1:
                    self.table[r["dialect"]][s[i1]][" ".join(t[j1:j2])] += 1
        self.best = {d: {w: c.most_common(1)[0][0] for w, c in tab.items()} for d, tab in self.table.items()}
        return self

    def translate(self, dialect, text):
        tab = self.best.get(dialect, {})
        return detokenize([tab.get(w, w) for w in tokenize(text)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--splits", nargs="*", default=["val", "test"])
    ap.add_argument("--extra_test", nargs="*", default=[], help="other test files, e.g. data/processed/onubad.jsonl")
    args = ap.parse_args()
    data = Path(args.data_dir)
    train = load_jsonl(data / "train.jsonl")
    dic = Dictionary().fit(train)
    print("dictionary entries per dialect:", {d: len(t) for d, t in dic.best.items()})

    targets = [(sp, data / f"{sp}.jsonl") for sp in args.splits] + [(Path(p).stem, Path(p)) for p in args.extra_test]
    for name, path in targets:
        rows = load_jsonl(path)
        out = Path(args.out_dir)
        write_jsonl(out / "copy" / "run1" / f"{name}.jsonl", [{**r, "hyp": r["src"]} for r in rows])
        write_jsonl(out / "dictionary" / "run1" / f"{name}.jsonl",
                    [{**r, "hyp": dic.translate(r["dialect"], r["src"])} for r in rows])
        print(f"wrote copy and dictionary outputs for {name} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

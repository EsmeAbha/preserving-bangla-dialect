"""Cheap CPU versions of the RQ3 and data-size experiments, using the dictionary baseline.

dictionary_pooled            one table learned from all five dialects (ignores the dialect label)
dictionary_lodo_<dialect>    pooled table learned from the other four dialects only, so the held-out
                             dialect is never seen; evaluate.py reads these as RQ3 folds
dictionary_frac<k>           per-dialect table learned from k% of the training groups (3 samples each)

  python src/dictionary_experiments.py --splits val
  python src/evaluate.py --split val --results_dir results/dict_ablations --main_system dictionary \
      --systems copy dictionary dictionary_pooled dictionary_frac10 ... dictionary_lodo_chittagong ...
"""
import argparse
import random
from collections import defaultdict
from pathlib import Path

from baselines import Dictionary
from common import DIALECTS, load_jsonl, write_jsonl


def pooled(rows):
    return [{**r, "dialect": "all"} for r in rows]


def translate(dic, rows, key=None):
    return [{**r, "hyp": dic.translate(key or r["dialect"], r["src"])} for r in rows]


def subsample(train, frac, seed):
    groups = defaultdict(list)
    for r in train:
        groups[r["group_id"]].append(r)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    keep = keys[: max(1, round(frac * len(keys)))]
    return [r for k in keep for r in groups[k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--splits", nargs="*", default=["val"])
    ap.add_argument("--extra_test", nargs="*", default=[])
    ap.add_argument("--fracs", nargs="*", type=float, default=[0.1, 0.25, 0.5])
    args = ap.parse_args()
    data, out = Path(args.data_dir), Path(args.out_dir)
    train = load_jsonl(data / "train.jsonl")
    targets = [(sp, data / f"{sp}.jsonl") for sp in args.splits] + [(Path(p).stem, Path(p)) for p in args.extra_test]

    pool = Dictionary().fit(pooled(train))
    lodo = {d: Dictionary().fit(pooled([r for r in train if r["dialect"] != d])) for d in DIALECTS}
    fracs = {(f, s): Dictionary().fit(subsample(train, f, s)) for f in args.fracs for s in (1, 2, 3)}

    for name, path in targets:
        rows = load_jsonl(path)
        write_jsonl(out / "dictionary_pooled" / "run1" / f"{name}.jsonl", translate(pool, rows, "all"))
        for d, dic in lodo.items():
            write_jsonl(out / f"dictionary_lodo_{d}" / "run1" / f"{name}.jsonl", translate(dic, rows, "all"))
        for (f, s), dic in fracs.items():
            write_jsonl(out / f"dictionary_frac{round(100 * f)}" / f"s{s}" / f"{name}.jsonl", translate(dic, rows))
        print(f"wrote dictionary experiments for {name} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

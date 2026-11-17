"""Builds a small fake Vashantor-like CSV folder to smoke-test the CPU pipeline (no real data, no GPU).

  python tests/make_synthetic.py --out /tmp/fake_raw
It also plants a cross-split leak (same standard sentence in train and test, one with a final "।")
so that the alignment step must move pairs.
"""
import argparse
import random
from pathlib import Path

import pandas as pd

STD = ["আজকে", "আমার", "মন", "ভালো", "নেই", "তুমি", "কোথায়", "যাচ্ছ", "আমি", "বাড়ি", "যাব", "সে", "ভাত", "খেয়েছে",
       "কি", "করছ", "এখন", "আমরা", "কাল", "আসব"]
SUB = {  # per-dialect word substitutions
    "chittagong": {"আজকে": "আযিয়া", "আমার": "আর", "ভালো": "বালা", "নেই": "নাই", "তুমি": "তুই", "কি": "কি্ত"},
    "noakhali": {"আজকে": "আইজ্জা", "আমার": "আর", "ভালো": "ভালা", "নেই": "নাই", "যাব": "যামু"},
    "sylhet": {"আজকে": "আইজকু", "ভালো": "ভালা", "নেই": "নায়", "কোথায়": "কোয়াই", "আমি": "আমি"},
    "barishal": {"আজকে": "আইজ", "আমার": "মোর", "মন": "মনডা", "নেই": "না", "বাড়ি": "বাড়ি"},
    "mymensingh": {"আজকে": "আইজ", "ভালো": "ভালা", "নেই": "নাই", "যাব": "যাইয়াম"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200, help="standard sentences")
    args = ap.parse_args()
    rng = random.Random(0)
    sents = list(dict.fromkeys(" ".join(rng.sample(STD, rng.randint(3, 7))) for _ in range(args.n * 2)))[: args.n]
    n_tr, n_va = int(0.75 * len(sents)), int(0.1 * len(sents))
    split_of = {s: "Train" if i < n_tr else "Validation" if i < n_tr + n_va else "Test" for i, s in enumerate(sents)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for d, sub in SUB.items():
        for sp in ("Train", "Validation", "Test"):
            part = [s for s in sents if split_of[s] == sp]
            if sp == "Test" and d == "sylhet":
                part = part + [sents[0] + "।"]  # leak: a training sentence (with "।") in Sylhet test
            if sp == "Train":
                part = part + [part[0], ""]  # one duplicate and one empty pair
            rows = [{f"{d.title()} Bangla Speech": " ".join(sub.get(w, w) for w in s.split()) if s else "",
                     "Bangla Speech": s} for s in part]
            pd.DataFrame(rows).to_csv(out / f"{d.title()}_{sp}.csv", index=False)
    print(f"wrote {len(list(out.glob('*.csv')))} files to {out}")


if __name__ == "__main__":
    main()

"""Full fine-tuning of BanglaT5 (csebuetnlp/banglat5) as the smaller task-specific comparison.

Input: "<Dialect>: <dialect sentence>", target: standard Bangla. The BanglaT5 authors normalise text
with their normaliser (pip install git+https://github.com/csebuetnlp/normalizer); it is applied to
inputs when installed. Outputs are scored against the original references. The epoch with the
best validation chrF is kept. Training is in fp32 because T5 is unstable in fp16.

  python src/train_banglat5.py --seed 1        (also 2, 3)
Writes outputs/banglat5/s<seed>/{val,test}.jsonl and checkpoints/banglat5_s<seed>/log.json
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from sacrebleu.metrics import CHRF

from common import DIALECT_NAMES, DIALECTS, load_jsonl, stratified_subset, write_jsonl

try:
    from normalizer import normalize as bt5_normalize
except ImportError:
    bt5_normalize = None


def source(row, with_dialect=True):
    text = bt5_normalize(row["src"]) if bt5_normalize else row["src"]
    return f"{DIALECT_NAMES[row['dialect']]}: {text}" if with_dialect else text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="csebuetnlp/banglat5")
    ap.add_argument("--name", default="banglat5")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--holdout", choices=DIALECTS, default=None)
    ap.add_argument("--val_subset", type=int, default=500)
    ap.add_argument("--beams", type=int, default=1, help="1 = greedy, as for the LLMs")
    ap.add_argument("--predict_splits", nargs="*", default=["val", "test"])
    ap.add_argument("--extra_test", nargs="*", default=[])
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    args = ap.parse_args()

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = Path(args.ckpt_dir) / f"{args.name}_s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    data = Path(args.data_dir)
    train = load_jsonl(data / "train.jsonl")
    val = load_jsonl(data / "val.jsonl")
    if args.holdout:
        train = [r for r in train if r["dialect"] != args.holdout]
        val = [r for r in val if r["dialect"] != args.holdout]
    val_sel = stratified_subset(val, args.val_subset, seed=0)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = math.ceil(len(train) / args.batch_size) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)

    @torch.no_grad()
    def predict(rows, bs=64):
        model.eval()
        out = []
        for b in range(0, len(rows), bs):
            part = rows[b:b + bs]
            enc = tok([source(r) for r in part], return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_len).to(dev)
            gen = model.generate(**enc, max_new_tokens=128, num_beams=args.beams, do_sample=False)
            out += [{**r, "hyp": h.strip()} for r, h in zip(part, tok.batch_decode(gen, skip_special_tokens=True))]
        model.train()
        return out

    def chrf(preds):
        return CHRF().corpus_score([p["hyp"] for p in preds], [[p["tgt"] for p in preds]]).score

    best, log, t0 = -1.0, [], time.time()
    model.train()
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train)
        for b in range(0, len(train), args.batch_size):
            part = train[b:b + args.batch_size]
            enc = tok([source(r) for r in part], return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_len).to(dev)
            lab = tok(text_target=[r["tgt"] for r in part], return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_len)["input_ids"].to(dev)
            lab[lab == tok.pad_token_id] = -100
            loss = model(**enc, labels=lab).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        score = chrf(predict(val_sel))
        log.append({"epoch": epoch, "loss": round(loss.item(), 4), "val_chrf": round(score, 2),
                    "minutes": round((time.time() - t0) / 60, 1)})
        print(log[-1], flush=True)
        if score > best:
            best = score
            model.save_pretrained(run_dir / "best")
            tok.save_pretrained(run_dir / "best")

    model = AutoModelForSeq2SeqLM.from_pretrained(run_dir / "best").to(dev)
    json.dump({"args": vars(args), "best_val_chrf_subset": round(best, 2), "log": log,
               "normaliser": bt5_normalize is not None}, open(run_dir / "log.json", "w"), indent=2)
    targets = [(sp, data / f"{sp}.jsonl") for sp in args.predict_splits] + [(Path(p).stem, Path(p)) for p in args.extra_test]
    for split, path in targets:
        preds = predict(load_jsonl(path))
        write_jsonl(Path(args.out_dir) / args.name / f"s{args.seed}" / f"{split}.jsonl", preds)
        print(f"wrote {split}: chrF {chrf(preds):.2f}")


if __name__ == "__main__":
    main()

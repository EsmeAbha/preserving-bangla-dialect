"""QLoRA fine-tuning of an instruction-tuned LLM on Vashantor (dialect -> standard Bangla).

Settings from the proposal (defaults): 4-bit NF4 base, LoRA r=32, alpha=64, dropout 0.05 on all
linear layers, lr 2e-4 cosine, effective batch 16, 2 epochs, max length 256. The loss is on the
answer tokens only. Validation chrF (on a fixed stratified validation subset) is measured several
times per epoch and the adapter with the best value is kept. After training, the best adapter is
used to write predictions for the full validation and test splits.

Main runs:         python src/train_qlora.py --model qwen7b --seed 1      (also 2, 3; and --model llama8b)
Data ablation:     python src/train_qlora.py --model qwen7b --train_frac 0.25 --name qwen7b_frac25
Rank ablation:     python src/train_qlora.py --model qwen7b --rank 8 --alpha 16 --name qwen7b_r8
No dialect name:   python src/train_qlora.py --model qwen7b --no_dialect_name --name qwen7b_nodialect
Leave one out:     python src/train_qlora.py --model qwen7b --holdout chittagong --name qwen7b_lodo_chittagong

Outputs: checkpoints/<name>_s<seed>/best (adapter), .../log.json, and
         outputs/<name>/s<seed>/{val,test}.jsonl
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

from common import DIALECTS, build_messages, load_jsonl, stratified_subset, write_jsonl
from llm_utils import MODELS, compute_dtype, generate, load_tokenizer, prompt_text


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def subsample_by_group(rows, frac, seed):
    """Keep a fraction of standard sentences (with all their dialect versions)."""
    groups = sorted({r["group_id"] for r in rows})
    keep = set(random.Random(seed).sample(groups, max(1, round(frac * len(groups)))))
    return [r for r in rows if r["group_id"] in keep]


def encode(tok, row, with_dialect, max_len):
    """input_ids and labels; labels are -100 on the prompt so only the answer (and end-of-turn) is learned."""
    msgs = build_messages(row, 0, with_dialect)
    prompt = prompt_text(tok, msgs)
    full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": row["tgt"]}], tokenize=False)
    if not full.startswith(prompt):  # some templates differ; fall back to prompt + answer + eos
        full = prompt + row["tgt"] + tok.eos_token
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
    labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[len(p_ids):]
    return f_ids, labels


def batches(examples, batch_size, pad_id, rng):
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    for b in range(0, len(idx), batch_size):
        part = [examples[i] for i in idx[b:b + batch_size]]
        n = max(len(x[0]) for x in part)
        ids = torch.full((len(part), n), pad_id)
        lab = torch.full((len(part), n), -100)
        att = torch.zeros((len(part), n), dtype=torch.long)
        for j, (x, y) in enumerate(part):
            ids[j, :len(x)] = torch.tensor(x)
            lab[j, :len(y)] = torch.tensor(y)
            att[j, :len(x)] = 1
        yield ids, att, lab


def lora_state(model):
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items() if "lora_" in k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen7b")
    ap.add_argument("--name", default=None, help="run name; default <model>_qlora")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4, help="effective batch = batch_size * grad_accum")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--train_frac", type=float, default=1.0)
    ap.add_argument("--holdout", choices=DIALECTS, default=None, help="leave-one-dialect-out: no data from it")
    ap.add_argument("--no_dialect_name", action="store_true")
    ap.add_argument("--evals_per_epoch", type=int, default=2)
    ap.add_argument("--val_subset", type=int, default=250, help="validation rows used for model selection")
    ap.add_argument("--gen_batch_size", type=int, default=32)
    ap.add_argument("--predict_splits", nargs="*", default=["val", "test"])
    ap.add_argument("--extra_test", nargs="*", default=[])
    ap.add_argument("--max_steps", type=int, default=None, help="smoke test only")
    ap.add_argument("--no_4bit", action="store_true", help="full-precision LoRA (CPU smoke test only)")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, get_cosine_schedule_with_warmup

    set_seed(args.seed)
    name = args.name or f"{Path(args.model).name}_qlora"
    run_dir = Path(args.ckpt_dir) / f"{name}_s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_id = MODELS.get(args.model, args.model)
    data = Path(args.data_dir)
    with_dialect = not args.no_dialect_name

    train = load_jsonl(data / "train.jsonl")
    val = load_jsonl(data / "val.jsonl")
    if args.holdout:  # the held-out dialect contributes nothing, not even to model selection
        train = [r for r in train if r["dialect"] != args.holdout]
        val_sel = [r for r in val if r["dialect"] != args.holdout]
    else:
        val_sel = val
    if args.train_frac < 1:
        train = subsample_by_group(train, args.train_frac, seed=args.seed)
    val_sel = stratified_subset(val_sel, args.val_subset, seed=0)
    print(f"{name} seed {args.seed}: {len(train)} training pairs, {len(val_sel)} validation rows for selection")

    tok = load_tokenizer(model_id)
    examples = [encode(tok, r, with_dialect, args.max_len) for r in train]
    print("token length: mean %.0f, max %d" % (np.mean([len(e[0]) for e in examples]), max(len(e[0]) for e in examples)))

    cuda = torch.cuda.is_available()
    dtype = compute_dtype() if cuda else torch.float32
    if args.no_4bit:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map={"": 0} if cuda else None)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    else:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=dtype)
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant, dtype=dtype,
                                                     device_map={"": 0})
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(examples) / (args.batch_size * args.grad_accum))
    total = args.max_steps or math.ceil(steps_per_epoch * args.epochs)
    sched = get_cosine_schedule_with_warmup(opt, int(args.warmup_ratio * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda and dtype == torch.float16)
    eval_every = max(1, steps_per_epoch // args.evals_per_epoch)
    print(f"{total} optimiser steps, validation every {eval_every} steps, compute dtype {dtype}")

    def validate():
        model.eval()
        model.config.use_cache = True
        preds = generate(model, tok, val_sel, batch_size=args.gen_batch_size, with_dialect=with_dialect, progress=False)
        model.config.use_cache = False
        model.train()
        return CHRF().corpus_score([p["hyp"] for p in preds], [[p["tgt"] for p in preds]]).score

    log, best, best_state, step, t0 = [], -1.0, None, 0, time.time()
    rng = random.Random(args.seed)
    model.train()
    done, running, logged = False, 0.0, 0
    while not done:
        micro = 0
        opt.zero_grad(set_to_none=True)  # drop a partial accumulation left from the previous epoch
        for ids, att, lab in batches(examples, args.batch_size, tok.pad_token_id, rng):
            ids, att, lab = ids.to(model.device), att.to(model.device), lab.to(model.device)
            with torch.autocast("cuda", dtype=dtype, enabled=cuda):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / args.grad_accum
            scaler.scale(loss).backward()
            running += loss.item()
            micro += 1
            if micro % args.grad_accum:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            logged += 1
            if step % 20 == 0:
                print(f"step {step}/{total} loss {running / logged:.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
                running, logged = 0.0, 0
            if step % eval_every == 0 or step == total:
                score = validate()
                log.append({"step": step, "epoch": round(step / steps_per_epoch, 2), "val_chrf": round(score, 2),
                            "minutes": round((time.time() - t0) / 60, 1)})
                print("validation", log[-1], flush=True)
                if score > best:
                    best, best_state = score, lora_state(model)
                    model.save_pretrained(run_dir / "best")
            if step >= total:
                done = True
                break

    # restore the best adapter and predict
    model.load_state_dict(best_state, strict=False)
    json.dump({"args": vars(args), "best_val_chrf_subset": round(best, 2), "log": log,
               "train_pairs": len(train), "minutes": round((time.time() - t0) / 60, 1)},
              open(run_dir / "log.json", "w"), indent=2)
    model.eval()
    model.config.use_cache = True
    targets = [(sp, data / f"{sp}.jsonl") for sp in args.predict_splits] + [(Path(p).stem, Path(p)) for p in args.extra_test]
    for split, path in targets:
        rows = load_jsonl(path)
        preds = generate(model, tok, rows, batch_size=args.gen_batch_size, with_dialect=with_dialect)
        write_jsonl(Path(args.out_dir) / name / f"s{args.seed}" / f"{split}.jsonl", preds)
        print(f"wrote {split}: chrF {CHRF().corpus_score([p['hyp'] for p in preds], [[p['tgt'] for p in preds]]).score:.2f}")


if __name__ == "__main__":
    main()

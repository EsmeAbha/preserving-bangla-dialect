"""Zero-shot, 5-shot and fine-tuned (LoRA adapter) generation with an open-weight LLM.

Examples (Kaggle GPU):
  # zero-shot, three instruction paraphrases = three runs
  for p in 0 1 2; do python generate.py --model qwen7b --mode zeroshot --instruction $p --run p$p --splits val; done
  # 5-shot, three example draws
  for s in 1 2 3; do python generate.py --model qwen7b --mode fewshot --shot_seed $s --run s$s --splits val; done
  # a trained adapter
  python generate.py --model qwen7b --adapter checkpoints/qwen7b_qlora_s1/best --system qwen7b_qlora --run s1

Writes outputs/<system>/<run>/<split>.jsonl with fields id, dialect, src, tgt, raw, hyp.
"""
import argparse
import time
from pathlib import Path

from common import ShotSampler, load_jsonl, write_jsonl
from llm_utils import MODELS, generate, load_model, load_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen7b", help=f"one of {list(MODELS)} or a HF model id")
    ap.add_argument("--mode", choices=["zeroshot", "fewshot", "finetuned"], default="zeroshot")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (implies --mode finetuned)")
    ap.add_argument("--instruction", type=int, default=0, help="instruction paraphrase 0/1/2 (zero-shot runs)")
    ap.add_argument("--no_dialect_name", action="store_true")
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--shot_seed", type=int, default=1)
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--splits", nargs="*", default=["val"])
    ap.add_argument("--extra_test", nargs="*", default=[], help="e.g. data/processed/onubad.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="first N rows only (smoke test)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--system", default=None, help="output folder name; default <model>_<mode>")
    ap.add_argument("--run", default="run1")
    ap.add_argument("--no_4bit", action="store_true")
    args = ap.parse_args()
    if args.adapter:
        args.mode = "finetuned"
    model_id = MODELS.get(args.model, args.model)
    system = args.system or f"{Path(args.model).name}_{args.mode}"

    tok = load_tokenizer(model_id)
    model = load_model(model_id, adapter=args.adapter, four_bit=not args.no_4bit)
    data = Path(args.data_dir)
    sampler = ShotSampler(load_jsonl(data / "train.jsonl"), k=args.shots, seed=args.shot_seed) \
        if args.mode == "fewshot" else None

    targets = [(sp, data / f"{sp}.jsonl") for sp in args.splits] + [(Path(p).stem, Path(p)) for p in args.extra_test]
    for name, path in targets:
        rows = load_jsonl(path)[: args.limit]
        t0 = time.time()
        preds = generate(model, tok, rows, batch_size=args.batch_size, instruction_id=args.instruction,
                         with_dialect=not args.no_dialect_name, shot_sampler=sampler)
        out = Path(args.out_dir) / system / args.run / f"{name}.jsonl"
        write_jsonl(out, preds)
        print(f"{system}/{args.run}/{name}: {len(rows)} rows in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()

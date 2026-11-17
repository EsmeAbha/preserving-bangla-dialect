#!/usr/bin/env bash
# Week 10: leave-one-dialect-out folds (Qwen2.5-7B, 1 seed) and ONUBAD out-of-domain evaluation.
set -euo pipefail
cd "$(dirname "$0")/.."
for D in chittagong noakhali sylhet barishal mymensingh; do
  python train_qlora.py --model qwen7b --seed 1 --holdout $D --name qwen7b_lodo_$D --predict_splits val test
done
# ONUBAD (prepare first: python prepare_onubad.py --raw_dir data/raw_onubad)
O=data/processed/onubad.jsonl
python baselines.py --splits --extra_test $O
python generate.py --model qwen7b --mode zeroshot --run p0 --splits --extra_test $O     # contamination check
python generate.py --model qwen7b --adapter checkpoints/qwen7b_qlora_s1/best --system qwen7b_qlora --run s1 --splits --extra_test $O
python generate.py --model llama8b --adapter checkpoints/llama8b_qlora_s1/best --system llama8b_qlora --run s1 --splits --extra_test $O
python evaluate.py --split onubad

#!/usr/bin/env bash
# Week 9: BanglaT5 (3 seeds) and the Qwen2.5-7B ablations (1 seed each).
set -euo pipefail
cd "$(dirname "$0")/.."
for S in 1 2 3; do python train_banglat5.py --seed $S; done
python train_qlora.py --model qwen7b --seed 1 --train_frac 0.25 --name qwen7b_frac25
python train_qlora.py --model qwen7b --seed 1 --train_frac 0.50 --name qwen7b_frac50
python train_qlora.py --model qwen7b --seed 1 --rank 8 --alpha 16 --name qwen7b_r8
python train_qlora.py --model qwen7b --seed 1 --no_dialect_name --name qwen7b_nodialect
# zero-shot without the dialect name, as a reference for the prompt ablation
python generate.py --model qwen7b --mode zeroshot --no_dialect_name --system qwen7b_zeroshot_nodialect --run p0 --splits val test
python evaluate.py --split val

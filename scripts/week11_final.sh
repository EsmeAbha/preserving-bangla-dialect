#!/usr/bin/env bash
# Week 11: final TEST runs for prompting (fine-tuned models already wrote test predictions),
# Groq on the test subset, then all statistics, human-eval sheets and the error-analysis sample.
set -euo pipefail
cd "$(dirname "$0")/.."
for M in qwen7b llama8b; do
  for P in 0 1 2; do python src/generate.py --model $M --mode zeroshot --instruction $P --run p$P --splits test; done
  for S in 1 2 3; do python src/generate.py --model $M --mode fewshot --shot_seed $S --run s$S --splits test; done
done
python src/groq_generate.py --splits test --subset 500
bash reproduce_main_table.sh
python src/human_eval.py make --systems qwen7b_zeroshot qwen7b_fewshot qwen7b_qlora --runs p0 s1 s1
python src/error_analysis.py sample --system qwen7b_qlora --run s1 --n 60

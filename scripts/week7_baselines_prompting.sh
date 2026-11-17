#!/usr/bin/env bash
# Week 7: baselines and prompting on VALIDATION only (prompt and paraphrases are fixed here).
set -euo pipefail
cd "$(dirname "$0")/.."
python src/baselines.py --splits val test
for M in qwen7b llama8b; do
  for P in 0 1 2; do python src/generate.py --model $M --mode zeroshot --instruction $P --run p$P --splits val; done
  for S in 1 2 3; do python src/generate.py --model $M --mode fewshot --shot_seed $S --run s$S --splits val; done
done
# Groq 70B reference (needs GROQ_API_KEY); 500-pair stratified subset because of rate limits
python src/groq_generate.py --splits val --subset 500
python src/evaluate.py --split val

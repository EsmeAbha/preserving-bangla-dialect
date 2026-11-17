#!/usr/bin/env bash
# Week 8: QLoRA main runs, 3 seeds per model (about 1 GPU-hour each on a Kaggle T4).
# Kaggle sessions stop after 12 h: run one model per session, e.g. `bash week8_qlora_main.sh qwen7b`.
set -euo pipefail
cd "$(dirname "$0")/.."
MODELS=${1:-"qwen7b llama8b"}
for M in $MODELS; do
  for S in 1 2 3; do
    python src/train_qlora.py --model $M --seed $S --name ${M}_qlora
  done
done
python src/evaluate.py --split val

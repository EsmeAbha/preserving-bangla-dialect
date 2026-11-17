#!/usr/bin/env bash
# Rebuilds every result table from the saved predictions in outputs/ (CPU is enough; LaBSE runs
# faster on a GPU). To regenerate the predictions themselves, run scripts/week7..week11 in order.
set -euo pipefail
cd "$(dirname "$0")"
MAIN="copy dictionary banglat5 qwen7b_zeroshot qwen7b_fewshot qwen7b_qlora llama8b_zeroshot llama8b_fewshot llama8b_qlora llama70b_groq_zeroshot"
ABL="qwen7b_qlora qwen7b_frac25 qwen7b_frac50 qwen7b_r8 qwen7b_nodialect qwen7b_zeroshot_nodialect"
LODO="qwen7b_qlora qwen7b_lodo_chittagong qwen7b_lodo_noakhali qwen7b_lodo_sylhet qwen7b_lodo_barishal qwen7b_lodo_mymensingh"
python src/evaluate.py --split test --labse --systems $MAIN \
  --compare qwen7b_qlora:qwen7b_zeroshot qwen7b_qlora:qwen7b_fewshot qwen7b_qlora:llama8b_qlora \
            qwen7b_qlora:banglat5 llama8b_qlora:llama8b_fewshot qwen7b_qlora:llama70b_groq_zeroshot
python src/evaluate.py --split test --labse --systems $ABL --results_dir results/ablations
python src/evaluate.py --split test --labse --systems $LODO --results_dir results/lodo
if ls outputs/*/*/onubad.jsonl >/dev/null 2>&1; then python src/evaluate.py --split onubad --labse; fi

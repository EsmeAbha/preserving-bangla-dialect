#!/usr/bin/env bash
# Builds every table of report/report.tex from outputs/ (CPU systems). Usage: bash scripts/paper_tables.sh [FT]
# FT = fine-tuned LLM system to report (default qwen05b_lora, falls back to qwen05b_lora_ep05).
set -euo pipefail
cd "$(dirname "$0")/.."
FT=${1:-qwen05b_lora}
[ -f outputs/$FT/s1/test.jsonl ] || FT=qwen05b_lora_ep05
MAIN="copy dictionary dictionary_pooled banglat5_cpu1ep $FT"
# 0) validation (development numbers quoted in the text)
python src/evaluate.py --split val --labse --main_system $FT --systems copy dictionary dictionary_pooled banglat5_cpu1ep \
  qwen3b_zeroshot qwen3b_fewshot $FT --compare qwen3b_fewshot:qwen3b_zeroshot qwen3b_fewshot:copy qwen3b_zeroshot:copy \
  $FT:qwen3b_fewshot $FT:dictionary dictionary:dictionary_pooled banglat5_cpu1ep:dictionary
# 1) full test set
python src/evaluate.py --split test --labse --main_system $FT --systems $MAIN \
  --compare $FT:dictionary $FT:banglat5_cpu1ep banglat5_cpu1ep:dictionary dictionary:dictionary_pooled $FT:copy
# 2) the 500-pair test subset on which the prompted 3B model was run
python - "$FT" <<'PY'
import json, sys
from pathlib import Path
sub = {json.loads(l)["id"] for l in open("outputs/qwen3b_zeroshot_sub500/p1/test.jsonl")}
for f in Path("outputs").glob("*/*/test.jsonl"):
    sysname = f.parts[1].replace("_sub500", "")
    if sysname.startswith(("dictionary_frac", "dictionary_lodo")):
        continue
    out = Path("outputs_sub500") / sysname / f.parts[2] / "test.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [l for l in open(f) if json.loads(l)["id"] in sub]
    out.write_text("".join(rows), encoding="utf-8")
PY
python src/evaluate.py --split test --labse --out_dir outputs_sub500 --results_dir results/test_sub500 \
  --main_system $FT --systems copy dictionary banglat5_cpu1ep qwen3b_zeroshot qwen3b_fewshot $FT \
  --compare qwen3b_fewshot:qwen3b_zeroshot qwen3b_fewshot:copy qwen3b_zeroshot:copy $FT:qwen3b_fewshot $FT:qwen3b_zeroshot
# 3) leave-one-dialect-out (dictionary) on test
python src/evaluate.py --split test --results_dir results/dict_ablations --main_system dictionary \
  --systems copy dictionary dictionary_pooled dictionary_frac10 dictionary_frac25 dictionary_frac50 \
  dictionary_lodo_chittagong dictionary_lodo_noakhali dictionary_lodo_sylhet dictionary_lodo_barishal dictionary_lodo_mymensingh
# 4) out of domain (novel ONUBAD)
python src/evaluate.py --split onubad_novel --labse --results_dir results --main_system $FT \
  --systems copy dictionary banglat5_cpu1ep $FT --compare $FT:dictionary banglat5_cpu1ep:dictionary
# 5) error profiles
python src/auto_error_analysis.py --split test --systems copy dictionary banglat5_cpu1ep $FT --write results/test/error_profile.md
python src/auto_error_analysis.py --split test --out_dir outputs_sub500 --systems copy dictionary banglat5_cpu1ep qwen3b_zeroshot qwen3b_fewshot $FT --write results/test_sub500/error_profile.md

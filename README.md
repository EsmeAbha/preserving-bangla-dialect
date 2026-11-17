# CSC5515 G14 — Translating Bangla regional dialects into standard Bangla

QLoRA fine-tuning of Qwen2.5-7B-Instruct (main) and Llama-3.1-8B-Instruct on Vashantor, compared with
copy / dictionary baselines, zero- and 5-shot prompting, fully fine-tuned BanglaT5 and zero-shot
Llama-3.3-70B (Groq). Metrics: chrF (primary), BLEU, LaBSE similarity, human meaning ratings.

## Files

Python modules are in `src/`; run everything from the repository root (e.g. `python src/evaluate.py`).

| File (in `src/` unless a folder is given) | What it does |
|---|---|
| `prepare_data.py` | Cleans Vashantor and aligns the splits by standard sentence → `data/processed/{train,val,test}.jsonl`, `stats.json` |
| `prepare_onubad.py` | ONUBAD → `data/processed/onubad.jsonl` (out-of-domain test; Chittagong, Sylhet, Barishal) |
| `common.py` | Prompt (from the proposal) + 2 paraphrases, few-shot sampler, output clean-up |
| `baselines.py` | Copy and word-dictionary baselines (CPU) |
| `generate.py` | Zero-shot / 5-shot / fine-tuned-adapter generation, greedy, 128 new tokens (GPU) |
| `groq_generate.py` | Llama-3.3-70B zero-shot via Groq; key from `GROQ_API_KEY`; resumable |
| `train_qlora.py` | QLoRA training (also data-size, rank, no-dialect-name ablations and leave-one-dialect-out via `--holdout`) |
| `train_banglat5.py` | Full fine-tuning of BanglaT5 |
| `evaluate.py` | All metrics, mean ± std over runs, paired bootstrap CIs, RQ2 distance + Spearman, RQ3 table |
| `human_eval.py` | Blind rating sheets (150 + 50) and quadratic-weighted κ |
| `error_analysis.py` | Error sample sheet (≥ 50) and category summary |
| `reproduce_main_table.sh` | Rebuilds every table from `outputs/` |
| `scripts/week*.sh` | The run plan of the proposal, week by week |
| `kaggle_notebook.ipynb` | Setup + run cells for a Kaggle GPU session |
| `tests/` | Synthetic data and fake outputs for a CPU smoke test |

All predictions are saved as `outputs/<system>/<run>/<split>.jsonl` (fields `id, dialect, src, tgt, raw, hyp`).
Runs are `p0 p1 p2` (zero-shot instruction paraphrases) or `s1 s2 s3` (seeds / few-shot draws).

## 1. Data

**Already included:** `data/processed/` holds the cleaned, aligned Vashantor splits, `onubad.jsonl` and
`onubad_novel.jsonl` (ONUBAD pairs that do not overlap Vashantor training data), derived from the CC BY 4.0
datasets below. `docs/week7_data_report.md` summarises them with the copy and dictionary baselines on
validation. Re-create them with the steps below.

1. Download `Vashantor_CSV_Format.zip` from Mendeley Data (V2, doi:10.17632/bj5jgk878b.2, CC BY 4.0) and
   unzip it into `data/raw/`. Only Bangla-script files are used (Banglish/English files are skipped).
2. `pip install -r requirements.txt`
3. `python src/prepare_data.py --raw_dir data/raw --out_dir data/processed`

The script prints, for each dialect and split, the file and the two columns used — check them. If the
columns are not detected, pass `--src_col <dialect column> --tgt_col <standard Bangla column>`.

Splits are aligned by standard sentence. Pairs are grouped by their normalised standard sentence, ignoring
punctuation and spaces, so `… নেই` and `… নেই।` are one group. A group found in more than one split is
moved entirely to test (if any member is in test), otherwise to validation. `stats.json` reports the removed
empty or duplicate pairs, the moved pairs, split sizes, and the validation copy-baseline chrF per dialect
(dialect distance = 100 − chrF).

ONUBAD (Mendeley Data V2, doi:10.17632/6ft99kf89b.2): unzip into `data/raw_onubad/`, then
`python src/prepare_onubad.py --raw_dir data/raw_onubad`. Check the printed column choices. The script
also counts ONUBAD standard sentences that occur in the Vashantor training set.

## 2. Experiments (Kaggle, GPU T4 ×2 or P100; one GPU is used)

Upload `data/processed/` and this folder as Kaggle datasets, and add the secrets `HF_TOKEN` (needed for
Llama-3.1, which is gated: accept its licence on Hugging Face first) and `GROQ_API_KEY`. Then follow
`kaggle_notebook.ipynb` or run the weekly scripts:

| Week | Script | Runs |
|---|---|---|
| 7 | `scripts/week7_baselines_prompting.sh` | baselines; zero-shot (3 paraphrases) and 5-shot (3 draws) for Qwen and Llama on **val**; Groq on a 500-pair val subset |
| 8 | `scripts/week8_qlora_main.sh qwen7b` then `… llama8b` | 3 seeds each (~1 GPU-h per run) |
| 9 | `scripts/week9_banglat5_ablations.sh` | BanglaT5 × 3; 25%/50% data, rank 8, no dialect name |
| 10 | `scripts/week10_lodo_onubad.sh` | 5 leave-one-dialect-out folds; ONUBAD |
| 11 | `scripts/week11_final.sh` | prompting on **test**, Groq test subset, all tables, human-eval and error sheets |

Training settings (defaults of `train_qlora.py`, as in the proposal): 4-bit NF4, LoRA r=32, α=64,
dropout 0.05, all linear layers, lr 2e-4 cosine (3% warm-up), effective batch 16 (4 × 4 accumulation),
2 epochs, max length 256, loss on answer tokens only. Validation chrF on a fixed stratified 250-row
validation subset is measured twice per epoch and the best adapter is kept. With `--holdout <dialect>`,
that dialect is removed from training **and** from model selection. With `--train_frac`, whole standard-sentence
groups are sampled, so alignment is kept. T4 has no bf16, so the scripts use fp16 with loss scaling there.

The test split is only used for final numbers. The training scripts already write test predictions,
but checkpoint selection and prompt choice use validation only. Don't look at `evaluate.py --split test` before Week 11.

## 3. Results

```
bash reproduce_main_table.sh
```
writes `results/test/main_table.md` (RQ1), `significance.md` (paired bootstrap, 1,000 resamples),
`rq2.md` (dialect distance, copy WER, Spearman ρ with bootstrap CI), `results/lodo/test/rq3_lodo.md`
(RQ3), `results/ablations/test/main_table.md` and `results/onubad/main_table.md` (out-of-domain /
contamination check: compare zero-shot on Vashantor and ONUBAD). Empty or unusable outputs count as 0.

Human evaluation: `python src/human_eval.py make` → give `human_eval/sheet_annotator1.csv` (150 rows) to
yourself and `sheet_annotator2.csv` (50 rows) to the second annotator; fill the `rating` column (1–5;
scale in `human_eval.py`); then `python src/human_eval.py score` (needs `results/test/sentences.csv`).

Error analysis: `python src/error_analysis.py sample` → label the `category` column of
`error_analysis/errors_qwen7b_qlora_s1.csv` (word_sense, dropped, negation_tense, still_dialect, other),
then `python src/error_analysis.py summarize`.

## CPU smoke test (no data, no GPU)

```
export PYTHONPATH=$PWD/src
python tests/make_synthetic.py --out /tmp/g14/raw
python src/prepare_data.py --raw_dir /tmp/g14/raw --out_dir /tmp/g14/proc
python src/baselines.py --data_dir /tmp/g14/proc --out_dir /tmp/g14/out
python tests/fake_llm_outputs.py --data_dir /tmp/g14/proc --out_dir /tmp/g14/out
python src/evaluate.py --out_dir /tmp/g14/out --results_dir /tmp/g14/res --compare qwen7b_qlora:qwen7b_zeroshot
```
The fake LLM outputs are noise around the reference and must never be reported.

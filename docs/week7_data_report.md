# Week 7 data report (real data)

Generated from `data/processed/stats.json`, `data/processed/onubad.stats.json` and `results/` (validation
split and ONUBAD only — the Vashantor test split has not been scored).

Sources: Vashantor, Mendeley Data V2, doi:10.17632/bj5jgk878b.2 (`Vashantor_CSV_Format.zip`, SHA-256
5abb4372…6fc9919 — matches Mendeley); ONUBAD, Mendeley Data V2, doi:10.17632/6ft99kf89b.2 (`Sentence.xlsx` only).
Both CC BY 4.0. The processed files in `data/processed/` are derived from them.

## Vashantor after cleaning and alignment

| Split | Pairs | Per dialect | Standard sentences | Mean words (dialect / standard) |
|---|---|---|---|---|
| train | 9370 | 1874 | 1876 | 6.7 / 6.7 |
| val | 1255 | 251 | 250 | 5.5 / 5.5 |
| test | 1875 | 375 | 381 | 7.2 / 7.3 |

- Columns used: `<dialect>_bangla_speech` → `bangla_speech` (Banglish, English and `region_name` columns ignored).
- Empty pairs removed: 0; exact duplicate pairs removed: 0.
- **Alignment: 5 pairs moved** (one standard sentence, *আপনি কেমন আছেন?*, was in train and
  validation; its training copies were moved to validation, so train has 1,874 and validation 251 pairs per dialect).
- Group sizes: {'5': 2488, '10': 2, '4': 6, '3': 1, '1': 7, '2': 3}. Groups with fewer than 5 members are spelling variants of the same
  standard sentence in different dialect files (e.g. নিষেধ / নিশেধ, স্বামীরে / স্বামীর); each sits in one split.
- Near-copies: 5 validation and 5 test standard sentences have a character similarity ≥ 0.9 to a training sentence,
  but differ in a word (e.g. আমার / তোমার বন্ধুরা …). They are different sentences and are kept.

## Baselines on validation

| System | Runs | n | chrF Chit | chrF Noak | chrF Sylh | chrF Bari | chrF Myme | chrF all | BLEU all | LaBSE all |
|---|---|---|---|---|---|---|---|---|---|---|
| copy | 1 | 1255 | 26.00 | 37.56 | 39.79 | 42.61 | 58.67 | 40.95 | 10.43 | – |
| dictionary | 1 | 1255 | 59.66 | 65.82 | 65.49 | 76.01 | 87.96 | 70.97 | 44.78 | – |

## RQ2: dialect distance (validation)

| Dialect | Distance | Copy WER | dictionary chrF |
|---|---|---|---|
| chittagong | 74.00 | 90.81 | 59.66 |
| noakhali | 62.44 | 72.70 | 65.82 |
| sylhet | 60.21 | 73.93 | 65.49 |
| barishal | 57.39 | 65.10 | 76.01 |
| mymensingh | 41.33 | 45.69 | 87.96 |

Per-sentence Spearman ρ (distance vs. dictionary chrF, mean over runs, pooled, n=1255): -0.557, 95% bootstrap CI [-0.595, -0.515]

Farthest → closest dialect: chittagong, noakhali, sylhet, barishal, mymensingh

Hardest → easiest for dictionary: chittagong, sylhet, noakhali, barishal, mymensingh

The distance ranking matches the hypothesis that Chittagonian is farthest from standard Bangla.

## ONUBAD (out-of-domain test)

- 2796 pairs after removing 144 duplicates ({'chittagong': 935, 'sylhet': 931, 'barishal': 930}).
- **323 pairs have a standard sentence that also occurs in the
  Vashantor training set**, and 126 have a dialect sentence that does. These are not really
  out-of-domain, so `onubad_novel.jsonl` (2437 pairs, {'chittagong': 822, 'sylhet': 809, 'barishal': 806}) leaves them out. Report both.

All ONUBAD pairs:

| System | Runs | n | chrF Chit | chrF Sylh | chrF Bari | chrF all | BLEU all | LaBSE all |
|---|---|---|---|---|---|---|---|---|
| copy | 1 | 2796 | 29.29 | 39.11 | 43.46 | 37.32 | 9.60 | – |
| dictionary | 1 | 2796 | 51.34 | 63.95 | 67.07 | 60.81 | 33.30 | – |

Novel ONUBAD pairs only:

| System | Runs | n | chrF Chit | chrF Sylh | chrF Bari | chrF all | BLEU all | LaBSE all |
|---|---|---|---|---|---|---|---|---|
| copy | 1 | 2437 | 29.82 | 39.39 | 43.72 | 37.63 | 10.40 | – |
| dictionary | 1 | 2437 | 49.95 | 62.92 | 65.60 | 59.45 | 31.52 | – |

## BanglaT5 CPU check run (1 epoch, not the final run)

`train_banglat5.py --seed 1 --epochs 1` on CPU (63 min), to test the script on the real model and data.
The planned runs use 8 epochs × 3 seeds on a Kaggle GPU; these numbers are not for the report table.

| System | chrF Chit | Noak | Sylh | Bari | Myme | chrF all | BLEU | LaBSE |
|---|---|---|---|---|---|---|---|---|
| dictionary | 59.66 | 65.82 | 65.49 | 76.01 | 87.96 | 70.97 | 44.78 | 0.91 |
| BanglaT5, 1 epoch | 55.09 | 64.71 | 67.34 | 75.53 | 86.41 | 69.86 | 46.65 | 0.91 |

After one epoch BanglaT5 is 1.1 chrF below the dictionary (95% paired bootstrap CI [−2.18, −0.08]) and above it in
BLEU; no empty outputs. Its per-dialect order is exactly the distance order (Spearman ρ = −0.565 between
per-sentence distance and chrF, CI [−0.605, −0.522]).

## Qwen2.5-3B-Instruct zero-shot on validation (CPU, bf16)

The backup model from the proposal (the 7B does not fit in this machine's memory). Greedy decoding, 128 new
tokens, chat template, the three instruction paraphrases in `src/common.py` as the three runs.

| Instruction | chrF | BLEU | LaBSE | WER |
|---|---|---|---|---|
| p0 (proposal prompt) | 39.03 | 9.19 | 0.678 | 74.5 |
| p1 | 40.50 | 9.80 | 0.680 | 71.7 |
| p2 | 40.09 | 9.57 | 0.676 | 72.8 |
| mean ± std | 39.87 ± 0.76 | 9.52 | 0.68 | |
| copy baseline | 40.95 | 10.43 | 0.67 | |

Per dialect (mean of 3): Chittagong 28.0, Noakhali 35.3, Sylhet 40.5, Barishal 40.5, Mymensingh 55.0.

- Zero-shot prompting of the 3B model is **below the copy baseline** (−1.08 chrF, 95% paired bootstrap CI
  [−1.61, −0.55]); it changes words but rarely produces standard forms, and hurts most on Mymensingh (55.0 vs. 58.7).
- The instruction wording matters little (1.5 chrF spread). No empty outputs.
- Per-sentence Spearman ρ between dialect distance and output chrF is −0.90 (CI [−0.911, −0.886]): the outputs stay
  close to the input, so their quality follows the distance almost exactly.

## Dictionary ablations and leave-one-dialect-out (validation, CPU)

`src/dictionary_experiments.py`; full tables in `results/dict_ablations/val/`. These give the non-LLM
reference points for RQ3 and the data-size ablation.

**Pooled vs. per-dialect table.** One table learned from all dialects scores 70.89 chrF against 70.97 for the
per-dialect tables (difference +0.07, CI [−0.40, +0.56]): dialect-specific tables do not help on average.

**Training-data size** (per-dialect table, share of training groups, 3 samples each):

| Training data | 10% | 25% | 50% | 100% |
|---|---|---|---|---|
| chrF | 59.03 ± 0.87 | 63.82 ± 0.99 | 67.26 ± 0.37 | 70.97 |

The curve is still rising at 100%, so more data would still help the dictionary.

**Leave one dialect out** (pooled table learned from the other four dialects, chrF on the held-out dialect):

| Held-out dialect | Copy | Unseen (other four) | Seen (full dictionary) | Gain over copy when unseen |
|---|---|---|---|---|
| Chittagong | 26.00 | 33.69 | 59.66 | +7.7 |
| Noakhali | 37.56 | 50.59 | 65.82 | +13.0 |
| Sylhet | 39.79 | 52.64 | 65.49 | +12.9 |
| Barishal | 42.61 | 56.25 | 76.01 | +13.6 |
| Mymensingh | 58.67 | 70.63 | 87.96 | +12.0 |

Words learned from the other dialects recover 40–50% of the gap between copy and the seen-dialect
dictionary, except for Chittagong, the most distant dialect, where only +7.7 transfers. This is the
reference that the Qwen-7B LODO folds (Week 9) should be compared with.

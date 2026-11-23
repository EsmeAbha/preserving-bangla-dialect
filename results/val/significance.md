# Paired bootstrap on val (1000 resamples, chrF, mean over runs)

| A | B | A − B | 95% CI | n | p (one-sided) |
|---|---|---|---|---|---|
| qwen3b_fewshot | qwen3b_zeroshot | +1.14 | [+0.60, +1.65] | 1255 | 0.000 |
| qwen3b_fewshot | copy | +0.06 | [-0.61, +0.74] | 1255 | 0.407 |
| qwen3b_zeroshot | copy | -1.08 | [-1.61, -0.55] | 1255 | 0.000 |
| qwen05b_lora | qwen3b_fewshot | +25.26 | [+23.90, +26.52] | 1255 | 0.000 |
| qwen05b_lora | dictionary | -4.69 | [-5.91, -3.62] | 1255 | 0.000 |
| dictionary | dictionary_pooled | +0.07 | [-0.40, +0.56] | 1255 | 0.399 |
| banglat5_cpu1ep | dictionary | -1.11 | [-2.18, -0.08] | 1255 | 0.020 |

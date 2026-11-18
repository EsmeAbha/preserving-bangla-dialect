# Paired bootstrap on val (1000 resamples, chrF, mean over runs)

| A | B | A − B | 95% CI | n | p (one-sided) |
|---|---|---|---|---|---|
| qwen3b_fewshot | qwen3b_zeroshot | +1.14 | [+0.60, +1.65] | 1255 | 0.000 |
| qwen3b_fewshot | copy | +0.06 | [-0.61, +0.74] | 1255 | 0.407 |
| qwen3b_fewshot | dictionary | -29.95 | [-31.13, -28.84] | 1255 | 0.000 |
| qwen3b_zeroshot | copy | -1.08 | [-1.61, -0.55] | 1255 | 0.000 |
| banglat5_cpu1ep | dictionary | -1.11 | [-2.18, -0.08] | 1255 | 0.020 |

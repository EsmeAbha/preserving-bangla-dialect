# Paired bootstrap on test (1000 resamples, chrF, mean over runs)

| A | B | A − B | 95% CI | n | p (one-sided) |
|---|---|---|---|---|---|
| qwen05b_lora | dictionary | -2.80 | [-3.59, -1.96] | 1875 | 0.000 |
| qwen05b_lora | banglat5_cpu1ep | -3.45 | [-4.27, -2.59] | 1875 | 0.000 |
| banglat5_cpu1ep | dictionary | +0.64 | [-0.17, +1.45] | 1875 | 0.051 |
| dictionary | dictionary_pooled | -1.30 | [-1.62, -0.94] | 1875 | 0.000 |
| qwen05b_lora | copy | +23.78 | [+22.86, +24.58] | 1875 | 0.000 |

# Paired bootstrap on test (1000 resamples, chrF, mean over runs)

| A | B | A − B | 95% CI | n | p (one-sided) |
|---|---|---|---|---|---|
| qwen05b_lora_ep05 | dictionary | -10.89 | [-11.68, -10.10] | 1875 | 0.000 |
| qwen05b_lora_ep05 | banglat5_cpu1ep | -11.54 | [-12.39, -10.63] | 1875 | 0.000 |
| banglat5_cpu1ep | dictionary | +0.64 | [-0.17, +1.45] | 1875 | 0.051 |
| dictionary | dictionary_pooled | -1.30 | [-1.62, -0.94] | 1875 | 0.000 |
| qwen05b_lora_ep05 | copy | +15.69 | [+14.83, +16.45] | 1875 | 0.000 |

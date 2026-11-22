# Paired bootstrap on test (1000 resamples, chrF, mean over runs)

| A | B | A − B | 95% CI | n | p (one-sided) |
|---|---|---|---|---|---|
| qwen3b_fewshot | qwen3b_zeroshot | +0.45 | [-0.53, +1.40] | 500 | 0.168 |
| qwen3b_fewshot | copy | -0.63 | [-1.71, +0.44] | 500 | 0.104 |
| qwen3b_zeroshot | copy | -1.08 | [-1.81, -0.37] | 500 | 0.001 |
| qwen05b_lora_ep05 | qwen3b_fewshot | +15.19 | [+13.51, +16.81] | 500 | 0.000 |
| qwen05b_lora_ep05 | qwen3b_zeroshot | +15.64 | [+14.16, +17.18] | 500 | 0.000 |

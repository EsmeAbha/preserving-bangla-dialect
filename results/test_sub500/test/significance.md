# Paired bootstrap on test (1000 resamples, chrF, mean over runs)

| A | B | A − B | 95% CI | n | p (one-sided) |
|---|---|---|---|---|---|
| qwen3b_fewshot | qwen3b_zeroshot | +0.45 | [-0.53, +1.40] | 500 | 0.168 |
| qwen3b_fewshot | copy | -0.63 | [-1.71, +0.44] | 500 | 0.104 |
| qwen3b_zeroshot | copy | -1.08 | [-1.81, -0.37] | 500 | 0.001 |
| qwen05b_lora | qwen3b_fewshot | +23.66 | [+21.90, +25.50] | 500 | 0.000 |
| qwen05b_lora | qwen3b_zeroshot | +24.11 | [+22.49, +25.91] | 500 | 0.000 |

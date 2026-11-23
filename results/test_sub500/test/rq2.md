# RQ2: dialect distance on test

Distance = 100 − chrF of the copy baseline; copy WER as a check.

| Dialect | Distance | Copy WER | qwen05b_lora chrF |
|---|---|---|---|
| chittagong | 68.03 | 85.92 | 55.26 |
| noakhali | 60.88 | 72.62 | 62.22 |
| sylhet | 47.42 | 52.68 | 67.11 |
| barishal | 60.03 | 65.99 | 70.23 |
| mymensingh | 37.71 | 41.84 | 86.01 |

Per-sentence Spearman ρ (distance vs. qwen05b_lora chrF, mean over runs, pooled, n=500): -0.514, 95% bootstrap CI [-0.577, -0.446]

Farthest → closest dialect: chittagong, noakhali, barishal, sylhet, mymensingh

Hardest → easiest for qwen05b_lora: chittagong, noakhali, sylhet, barishal, mymensingh

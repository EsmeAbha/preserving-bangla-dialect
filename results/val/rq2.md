# RQ2: dialect distance on val

Distance = 100 − chrF of the copy baseline; copy WER as a check.

| Dialect | Distance | Copy WER | qwen3b_fewshot chrF |
|---|---|---|---|
| chittagong | 74.00 | 90.81 | 29.63 |
| noakhali | 62.44 | 72.70 | 36.84 |
| sylhet | 60.21 | 73.93 | 41.38 |
| barishal | 57.39 | 65.10 | 41.53 |
| mymensingh | 41.33 | 45.69 | 55.56 |

Per-sentence Spearman ρ (distance vs. qwen3b_fewshot chrF, mean over runs, pooled, n=1255): -0.815, 95% bootstrap CI [-0.836, -0.791]

Farthest → closest dialect: chittagong, noakhali, sylhet, barishal, mymensingh

Hardest → easiest for qwen3b_fewshot: chittagong, noakhali, sylhet, barishal, mymensingh

# RQ2: dialect distance on val

Distance = 100 − chrF of the copy baseline; copy WER as a check.

| Dialect | Distance | Copy WER | qwen3b_zeroshot chrF |
|---|---|---|---|
| chittagong | 74.00 | 90.81 | 27.98 |
| noakhali | 62.44 | 72.70 | 35.28 |
| sylhet | 60.21 | 73.93 | 40.54 |
| barishal | 57.39 | 65.10 | 40.46 |
| mymensingh | 41.33 | 45.69 | 54.97 |

Per-sentence Spearman ρ (distance vs. qwen3b_zeroshot chrF, mean over runs, pooled, n=1255): -0.900, 95% bootstrap CI [-0.911, -0.886]

Farthest → closest dialect: chittagong, noakhali, sylhet, barishal, mymensingh

Hardest → easiest for qwen3b_zeroshot: chittagong, noakhali, barishal, sylhet, mymensingh

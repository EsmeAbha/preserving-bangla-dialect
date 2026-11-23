# RQ2: dialect distance on val

Distance = 100 − chrF of the copy baseline; copy WER as a check.

| Dialect | Distance | Copy WER | qwen05b_lora chrF |
|---|---|---|---|
| chittagong | 74.00 | 90.81 | 55.38 |
| noakhali | 62.44 | 72.70 | 63.40 |
| sylhet | 60.21 | 73.93 | 61.58 |
| barishal | 57.39 | 65.10 | 67.89 |
| mymensingh | 41.33 | 45.69 | 83.11 |

Per-sentence Spearman ρ (distance vs. qwen05b_lora chrF, mean over runs, pooled, n=1255): -0.445, 95% bootstrap CI [-0.491, -0.395]

Farthest → closest dialect: chittagong, noakhali, sylhet, barishal, mymensingh

Hardest → easiest for qwen05b_lora: chittagong, sylhet, noakhali, barishal, mymensingh

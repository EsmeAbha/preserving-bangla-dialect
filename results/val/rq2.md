# RQ2: dialect distance on val

Distance = 100 − chrF of the copy baseline; copy WER as a check.

| Dialect | Distance | Copy WER | qwen05b_lora_ep05 chrF |
|---|---|---|---|
| chittagong | 74.00 | 90.81 | 45.78 |
| noakhali | 62.44 | 72.70 | 56.44 |
| sylhet | 60.21 | 73.93 | 57.41 |
| barishal | 57.39 | 65.10 | 60.35 |
| mymensingh | 41.33 | 45.69 | 75.50 |

Per-sentence Spearman ρ (distance vs. qwen05b_lora_ep05 chrF, mean over runs, pooled, n=1255): -0.538, 95% bootstrap CI [-0.577, -0.495]

Farthest → closest dialect: chittagong, noakhali, sylhet, barishal, mymensingh

Hardest → easiest for qwen05b_lora_ep05: chittagong, noakhali, sylhet, barishal, mymensingh

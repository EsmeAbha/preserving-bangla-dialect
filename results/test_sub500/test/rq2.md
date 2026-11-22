# RQ2: dialect distance on test

Distance = 100 − chrF of the copy baseline; copy WER as a check.

| Dialect | Distance | Copy WER | qwen05b_lora_ep05 chrF |
|---|---|---|---|
| chittagong | 68.03 | 85.92 | 45.71 |
| noakhali | 60.88 | 72.62 | 53.81 |
| sylhet | 47.42 | 52.68 | 61.78 |
| barishal | 60.03 | 65.99 | 60.64 |
| mymensingh | 37.71 | 41.84 | 76.76 |

Per-sentence Spearman ρ (distance vs. qwen05b_lora_ep05 chrF, mean over runs, pooled, n=500): -0.569, 95% bootstrap CI [-0.628, -0.507]

Farthest → closest dialect: chittagong, noakhali, barishal, sylhet, mymensingh

Hardest → easiest for qwen05b_lora_ep05: chittagong, noakhali, barishal, sylhet, mymensingh

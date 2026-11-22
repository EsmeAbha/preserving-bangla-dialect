# RQ2: dialect distance on test

Distance = 100 − chrF of the copy baseline; copy WER as a check.

| Dialect | Distance | Copy WER | qwen05b_lora_ep05 chrF |
|---|---|---|---|
| chittagong | 69.03 | 86.07 | 44.47 |
| noakhali | 61.53 | 73.53 | 54.57 |
| sylhet | 48.52 | 56.18 | 61.73 |
| barishal | 61.84 | 68.59 | 59.38 |
| mymensingh | 40.59 | 43.88 | 76.88 |

Per-sentence Spearman ρ (distance vs. qwen05b_lora_ep05 chrF, mean over runs, pooled, n=1875): -0.546, 95% bootstrap CI [-0.581, -0.510]

Farthest → closest dialect: chittagong, barishal, noakhali, sylhet, mymensingh

Hardest → easiest for qwen05b_lora_ep05: chittagong, noakhali, barishal, sylhet, mymensingh

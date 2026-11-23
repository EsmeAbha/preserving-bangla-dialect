# RQ2: dialect distance on test

Distance = 100 − chrF of the copy baseline; copy WER as a check.

| Dialect | Distance | Copy WER | qwen05b_lora chrF |
|---|---|---|---|
| chittagong | 69.03 | 86.07 | 54.60 |
| noakhali | 61.53 | 73.53 | 61.63 |
| sylhet | 48.52 | 56.18 | 67.38 |
| barishal | 61.84 | 68.59 | 68.69 |
| mymensingh | 40.59 | 43.88 | 85.06 |

Per-sentence Spearman ρ (distance vs. qwen05b_lora chrF, mean over runs, pooled, n=1875): -0.492, 95% bootstrap CI [-0.528, -0.455]

Farthest → closest dialect: chittagong, barishal, noakhali, sylhet, mymensingh

Hardest → easiest for qwen05b_lora: chittagong, noakhali, sylhet, barishal, mymensingh

# RQ2: dialect distance on onubad_novel

Distance = 100 − chrF of the copy baseline; copy WER as a check.

| Dialect | Distance | Copy WER | qwen05b_lora chrF |
|---|---|---|---|
| chittagong | 70.18 | 82.91 | 49.20 |
| sylhet | 60.61 | 70.68 | 59.21 |
| barishal | 56.28 | 64.51 | 62.26 |

Per-sentence Spearman ρ (distance vs. qwen05b_lora chrF, mean over runs, pooled, n=2437): -0.420, 95% bootstrap CI [-0.455, -0.386]

Farthest → closest dialect: chittagong, sylhet, barishal

Hardest → easiest for qwen05b_lora: chittagong, sylhet, barishal

# RQ2: dialect distance on onubad_novel

Distance = 100 − chrF of the copy baseline; copy WER as a check.

| Dialect | Distance | Copy WER | qwen05b_lora_ep05 chrF |
|---|---|---|---|
| chittagong | 70.18 | 82.91 | 37.96 |
| sylhet | 60.61 | 70.68 | 49.90 |
| barishal | 56.28 | 64.51 | 53.75 |

Per-sentence Spearman ρ (distance vs. qwen05b_lora_ep05 chrF, mean over runs, pooled, n=2437): -0.507, 95% bootstrap CI [-0.537, -0.475]

Farthest → closest dialect: chittagong, sylhet, barishal

Hardest → easiest for qwen05b_lora_ep05: chittagong, sylhet, barishal

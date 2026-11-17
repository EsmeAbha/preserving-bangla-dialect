# RQ2: dialect distance on onubad_novel

Distance = 100 − chrF of the copy baseline; copy WER as a check.

| Dialect | Distance | Copy WER | dictionary chrF |
|---|---|---|---|
| chittagong | 70.18 | 82.91 | 49.95 |
| sylhet | 60.61 | 70.68 | 62.92 |
| barishal | 56.28 | 64.51 | 65.60 |

Per-sentence Spearman ρ (distance vs. dictionary chrF, mean over runs, pooled, n=2437): -0.588, 95% bootstrap CI [-0.617, -0.557]

Farthest → closest dialect: chittagong, sylhet, barishal

Hardest → easiest for dictionary: chittagong, sylhet, barishal

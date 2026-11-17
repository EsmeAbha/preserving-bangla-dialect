# RQ2: dialect distance on onubad

Distance = 100 − chrF of the copy baseline; copy WER as a check.

| Dialect | Distance | Copy WER | dictionary chrF |
|---|---|---|---|
| chittagong | 70.71 | 83.20 | 51.34 |
| sylhet | 60.89 | 70.75 | 63.95 |
| barishal | 56.54 | 64.20 | 67.07 |

Per-sentence Spearman ρ (distance vs. dictionary chrF, mean over runs, pooled, n=2796): -0.573, 95% bootstrap CI [-0.601, -0.544]

Farthest → closest dialect: chittagong, sylhet, barishal

Hardest → easiest for dictionary: chittagong, sylhet, barishal

# RQ2: dialect distance on test

Distance = 100 − chrF of the copy baseline; copy WER as a check.

| Dialect | Distance | Copy WER | dictionary chrF |
|---|---|---|---|
| chittagong | 69.03 | 86.07 | 56.12 |
| noakhali | 61.53 | 73.53 | 60.79 |
| sylhet | 48.52 | 56.18 | 68.48 |
| barishal | 61.84 | 68.59 | 78.22 |
| mymensingh | 40.59 | 43.88 | 87.90 |

Per-sentence Spearman ρ (distance vs. dictionary chrF, mean over runs, pooled, n=1875): -0.583, 95% bootstrap CI [-0.617, -0.550]

Farthest → closest dialect: chittagong, barishal, noakhali, sylhet, mymensingh

Hardest → easiest for dictionary: chittagong, noakhali, sylhet, barishal, mymensingh

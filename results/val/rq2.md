# RQ2: dialect distance on val

Distance = 100 − chrF of the copy baseline; copy WER as a check.

| Dialect | Distance | Copy WER | dictionary chrF |
|---|---|---|---|
| chittagong | 74.00 | 90.81 | 59.66 |
| noakhali | 62.44 | 72.70 | 65.82 |
| sylhet | 60.21 | 73.93 | 65.49 |
| barishal | 57.39 | 65.10 | 76.01 |
| mymensingh | 41.33 | 45.69 | 87.96 |

Per-sentence Spearman ρ (distance vs. dictionary chrF, mean over runs, pooled, n=1255): -0.557, 95% bootstrap CI [-0.595, -0.515]

Farthest → closest dialect: chittagong, noakhali, sylhet, barishal, mymensingh

Hardest → easiest for dictionary: chittagong, sylhet, noakhali, barishal, mymensingh

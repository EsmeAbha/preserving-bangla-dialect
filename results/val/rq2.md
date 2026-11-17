# RQ2: dialect distance on val

Distance = 100 − chrF of the copy baseline; copy WER as a check.

| Dialect | Distance | Copy WER | banglat5_cpu1ep chrF |
|---|---|---|---|
| chittagong | 74.00 | 90.81 | 55.09 |
| noakhali | 62.44 | 72.70 | 64.71 |
| sylhet | 60.21 | 73.93 | 67.34 |
| barishal | 57.39 | 65.10 | 75.53 |
| mymensingh | 41.33 | 45.69 | 86.41 |

Per-sentence Spearman ρ (distance vs. banglat5_cpu1ep chrF, mean over runs, pooled, n=1255): -0.565, 95% bootstrap CI [-0.605, -0.522]

Farthest → closest dialect: chittagong, noakhali, sylhet, barishal, mymensingh

Hardest → easiest for banglat5_cpu1ep: chittagong, noakhali, sylhet, barishal, mymensingh

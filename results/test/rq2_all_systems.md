# RQ2 on test: distance vs. difficulty for every system

Dialect distance: chittagong 69.0, noakhali 61.5, sylhet 48.5, barishal 61.8, mymensingh 40.6

| System | Sentence-level ρ | 95% CI | n | Dialect-level ρ (5 dialects) | Hardest → easiest |
|---|---|---|---|---|---|
| copy | -1.000 | [-1.000, -1.000] | 1875 | -1.00 | Chit, Bari, Noak, Sylh, Myme |
| dictionary | -0.583 | [-0.617, -0.550] | 1875 | -0.70 | Chit, Noak, Sylh, Bari, Myme |
| banglat5_cpu1ep | -0.579 | [-0.612, -0.544] | 1875 | -0.90 | Chit, Noak, Bari, Sylh, Myme |
| qwen05b_lora | -0.492 | [-0.528, -0.455] | 1875 | -0.70 | Chit, Noak, Sylh, Bari, Myme |

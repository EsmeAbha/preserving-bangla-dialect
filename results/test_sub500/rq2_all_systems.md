# RQ2 on test: distance vs. difficulty for every system

Dialect distance: chittagong 68.0, noakhali 60.9, sylhet 47.4, barishal 60.0, mymensingh 37.7

| System | Sentence-level ρ | 95% CI | n | Dialect-level ρ (5 dialects) | Hardest → easiest |
|---|---|---|---|---|---|
| copy | -1.000 | [-1.000, -1.000] | 500 | -1.00 | Chit, Noak, Bari, Sylh, Myme |
| dictionary | -0.626 | [-0.679, -0.569] | 500 | -0.90 | Chit, Noak, Sylh, Bari, Myme |
| banglat5_cpu1ep | -0.583 | [-0.644, -0.521] | 500 | -1.00 | Chit, Noak, Bari, Sylh, Myme |
| qwen3b_zeroshot | -0.900 | [-0.923, -0.870] | 500 | -1.00 | Chit, Noak, Bari, Sylh, Myme |
| qwen3b_fewshot | -0.769 | [-0.808, -0.726] | 500 | -1.00 | Chit, Noak, Bari, Sylh, Myme |
| qwen05b_lora | -0.514 | [-0.577, -0.446] | 500 | -0.90 | Chit, Noak, Sylh, Bari, Myme |

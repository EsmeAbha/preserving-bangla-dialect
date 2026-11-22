# RQ2 on test: distance vs. difficulty for every system

Dialect distance: chittagong 69.0, noakhali 61.5, sylhet 48.5, barishal 61.8, mymensingh 40.6

| System | Sentence-level ρ | 95% CI | n | Dialect-level ρ (5 dialects) | Hardest → easiest |
|---|---|---|---|---|---|
| dictionary | -0.583 | [-0.617, -0.550] | 1875 | -0.70 | Chit, Noak, Sylh, Bari, Myme |
| dictionary_pooled | -0.573 | [-0.608, -0.539] | 1875 | -0.70 | Chit, Noak, Sylh, Bari, Myme |
| banglat5_cpu1ep | -0.579 | [-0.612, -0.544] | 1875 | -0.90 | Chit, Noak, Bari, Sylh, Myme |
| qwen05b_lora_ep05 | -0.546 | [-0.581, -0.510] | 1875 | -0.90 | Chit, Noak, Bari, Sylh, Myme |

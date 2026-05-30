# E.4 (K, L) and accuracy

There are no naive relations between (K, L) and downstream accuracies since (K, L) not only influences sampling quality but also the computation budget. One safe way to discuss the relation between (K, L) and accuracy is: Fixing the computation budget, larger (K, L) will potentially produce higher accuracy, since the sampling quality is higher. Our experimental results show that,

• Increasing (K, L) can significantly improve accuracy in relatively longer contexts Table [10.](#page-22-1)

<span id="page-22-1"></span>Table 10 We show the effectiveness of larger hash tables for longer contexts by evaluating MegaBeam-Mistral-7B-512K on RULER [\(Hsieh et al.,](#page-13-11) [2024\)](#page-13-11). With the same computation cost (∼ 2%), config (11, 300) achieves higher accuracy compared to (10, 150).

| (K, L)            | 16K          | 128K         | 256K         |
|-------------------|--------------|--------------|--------------|
| Full<br>(10, 150) | 91.7<br>89.8 | 83.7<br>80.7 | 82.5<br>79.0 |
| (11, 300)         | 90.6         | 83.3         | 81.9         |

• Same set of (K, L) can generalize to larger LLMs Table [11.](#page-22-2)

<span id="page-22-2"></span>Table 11 8B and 70B models on RULER [\(Hsieh et al.,](#page-13-11) [2024\)](#page-13-11) 64K.

| Models/Config          | Full | (10, 150) | (10, 135) | (9, 120) | (9, 110) |
|------------------------|------|-----------|-----------|----------|----------|
| Llama-3.1-8B-Instruct  | 86.1 | 84.8      | 83.6      | 84.7     | 84.7     |
| Llama-3.1-70B-Instruct | 89.2 | 87.5      | 86.7      | 88.8     | 88.4     |


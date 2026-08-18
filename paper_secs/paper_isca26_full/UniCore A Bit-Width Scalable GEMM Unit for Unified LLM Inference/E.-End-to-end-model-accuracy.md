# E. End-to-end model accuracy

Tables I–II summarize WikiText-2 [29] perplexity and zero-shot accuracy [4], [8], [37], [47] for UNICORE and prior accelerators under several W/A/KV bit configurations on OPT-6.7B,

<span id="page-11-2"></span>TABLE II: Zero-shot performance on four benchmark datasets. Higher scores indicate better accuracy. UNICORE-Q enables distribution-adaptive DynFP quantization.

| Model        | Method    | Bits     | ARC-e | Hella. | Piqa  | Wino. | Avg.(↑) |
|--------------|-----------|----------|-------|--------|-------|-------|---------|
| WIOGCI       |           |          |       |        |       | 73.09 |         |
| Llama-3-8B   | FP16      | 16/16/16 | 77.82 | 79.25  | 80.79 |       | 77.74   |
|              | INT       | 8/8/16   | 77.36 | 78.99  | 79.98 | 73.64 | 77.49   |
|              | UniCore   | 8/8/16   | 77.61 | 79.09  | 80.41 | 73.24 | 77.59   |
|              | INT       | 4/4/16   | 75.13 | 76.78  | 79.98 | 70.09 | 75.49   |
| Liailia-3-6D | M-ANT     | 4/4/16   | 73.48 | 76.99  | 78.94 | 70.09 | 74.88   |
|              | BitMoD    | 4/4/16   | 73.65 | 76.75  | 78.40 | 69.77 | 74.64   |
|              | UNICORE   | 4/4/16   | 73.61 | 76.66  | 78.24 | 70.88 | 74.85   |
|              | UniCore-Q | 4/4/16   | 75.21 | 77.72  | 79.16 | 70.64 | 75.68   |
| Qwen3-8B     | FP16      | 16/16/16 | 80.64 | 74.94  | 77.37 | 67.72 | 75.17   |
|              | INT       | 8/8/16   | 79.34 | 75.06  | 77.48 | 67.80 | 74.92   |
|              | UniCore   | 8/8/16   | 80.72 | 74.53  | 77.53 | 67.88 | 75.16   |
|              | INT       | 4/4/16   | 73.82 | 71.53  | 75.24 | 65.98 | 71.64   |
|              | M-ANT     | 4/4/16   | 76.05 | 71.90  | 74.97 | 65.82 | 72.19   |
|              | BitMoD    | 4/4/16   | 78.03 | 71.41  | 74.97 | 65.19 | 72.40   |
|              | UNICORE   | 4/4/16   | 78.24 | 72.74  | 75.90 | 66.85 | 73.43   |
|              | UniCore-Q | 4/4/16   | 78.28 | 73.01  | 76.01 | 66.30 | 73.40   |

Llama-2-7B, Llama-3-8B, and Qwen3-8B/14B. All methods use direct-cast quantization without calibration, and FP16 serves as the accuracy reference. UNICORE-Q additionally uses offline weight-format search and online K/V format selection when K/V quantization is enabled.

1) Perplexity: As shown in Table I, in WikiText-2 perplexity, existing 4-bit formats (INT, MXFP4, BitMoD, and M-ANT) already show noticeable degradation at 4/4/16, with higher perplexity than FP16 across all models, whereas UNI-CORE-Q consistently achieves the lowest perplexity in each 4/4/16 column (e.g., 10.93 vs. 11.08–11.26 on OPT-6.7B, with similar gaps on Llama-2/3 and Qwen3). At 4/8/16 and 3/8/16, UNICORE-Q maintains more stable perplexity than other schemes and attains the best or near-best perplexity in almost all cases; the 4/4/4 setting follows the same trend. The few exceptions mainly arise when Owen3's smoother activations better match INT activation quantization, or when outlier groups/layers trigger PPL spikes. Compared with fixed-width mixed-precision designs under the 4/16/16 setting (AxCore), UNICORE-Q achieves lower perplexity on most models even with lower activation precision (e.g., 4/8/16). Overall, across nearly all models ranging from 6.7B to 70B parameters and bit-width combinations, the UNICORE family achieves leading perplexity among quantized schemes, while competing formats incur larger degradation or even become unstable in the most aggressive regimes.

2) Zero-shot Performance: The zero-shot results in Table II exhibit a similar pattern. Across both models, the UNICORE family consistently matches or surpasses most low-bit baselines and often approaches FP16 performance. At the 8/8/16 setting, UNICORE achieves leading average accuracy among quantized methods. Under the more aggressive 4/4/16 configuration, the UNICORE family continues to outperform INT, M-ANT, and BitMoD on most tasks, with DynFP (UNICORE-Q) providing further gains in most cases and helping the family achieve the best 4-bit average accuracy across both evaluated models. Minor task-level fluctuations occur because small quantization-induced logit shifts can flip discrete zero-shot predictions. Overall, UNICORE variants deliver state-of-

<span id="page-11-0"></span>![](_page_11_Figure_5.jpeg)

Fig. 19: Normalized energy and speedup of UNICORE compared with baselines in decode phase equipped with DDR4.

<span id="page-11-1"></span>![](_page_11_Figure_7.jpeg)

Fig. 20: Normalized energy and speedup of UNICORE compared with baselines in decode phase equipped with HBM2.

the-art zero-shot performance in the low-bit regime, with clear gains from DynFP-based weight adaptation.


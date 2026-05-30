# A.11 CHALLENGING BENCHMARKS

We carried out ablation studies between our method and diverse baselines on more arduous tasks shown on Tab[.17.](#page-22-0) Main tasks include: MMLU[\(Wang et al., 2024\)](#page-11-15), MathQA[\(Amini et al., 2019\)](#page-9-7), GSM8K[\(Cobbe et al., 2021\)](#page-9-8), HumanEval[\(Chen et al., 2021\)](#page-9-9).

<span id="page-22-0"></span>Model Bits Method MMLU MathQA GSM8K HumanEval 16 FP16 59.60 37.55 62.55 32.32 GPTO 26.94 19.33 15.42 13.88 23.21 Qwen1.5-MoE-A2.7B MoeQuant 34.75 22.42 18.12 VQ 48.79 28.34 46.51 28.03 KBVQ-MoE 52.16 30.75 56.39 30.11 79.5 16 FP16 58.53 85.44 71.5 **GPTQ** 42.11 34.9 49.34 50.12 Qwen3-30B-A3B MoeQuant 45.1 39.86 57.38 55.32 2 VQ 49.87 60.94 64.83 61.2 KBVQ-MoE 68.19 50.11 78.92 66.89 44.60 20.16 16 FP16 31.49 26.83 **GPTQ** 28.48 18.82 11.26 14.87 DeepSeekMoE-16B MoeQuant 34.82 24.93 12.88 16.95 2 30.78 24.49 13.55 17.89 VQ KBVQ-MoE 41.39 28.97 17.32 22.62 FP16 70.50 42.41 32.93 16 65.88 9.97 **GPTQ** 22.96 40.80 5.89 Mixtral-7x8B MoeQuant 49.39 27.11 20.67 14.74 2 VQ 55.84 30.83 49.4 23.21 KBVQ-MoE 29.9 61.11 34.8 57.86

Table 17: more challenging benchmarks

### A.12 LIMITATIONS

Despite the strong empirical performance of KBVQ-MoE on ultra-low-bit quantization of decoderonly MoE large language models, the method exhibits several limitations that highlight potential directions for future work.

- (1) Dependence on empirically selected SVD truncation rank. The truncated rank (k) in IDRE is currently chosen based on an empirical balance between reconstruction fidelity and storage overhead (e.g.,  $k=\frac{1}{128}$  of the full rank in our experiments). Although this choice works robustly across evaluated models, KBVQ-MoE does not yet include an adaptive mechanism to automatically determine the optimal rank for different MoE architectures, task regimes, or input statistics. This limits its ability to fully optimize the trade-off between redundancy removal and quantization efficiency across diverse settings.
- (2) Limited validation beyond decoder-only architectures. Our experiments focus primarily on decoder-only MoE models such as Qwen-MoE and Mixtral-8×7B. While this allows for a clean examination of the effects of IDRE and BCOS, the method has not yet been systematically evaluated on encoder-decoder MoE models or multimodal MoE architectures. These settings may require modified input-statistics estimation or revised forms of redundancy modeling. Extending KBVQ-MoE to bidirectional or cross-modal MoE structures is therefore an important area for future exploration.
- (3) Lack of evaluation in extreme bit regimes. KBVQ-MoE achieves near–FP16 accuracy at 2–3 bits, but has not been tested in more extreme quantization regimes such as 1-bit binary quantization or hybrid bit-widths (e.g., 1.5-bit). These regimes pose substantially greater information bottlenecks, and may require enhanced bias-correction mechanisms, more expressive codebook structures, or hybrid compression strategies to remain effective.

Together, these limitations delineate the boundaries of the current investigation and point toward promising extensions of KBVQ-MoE in future research.
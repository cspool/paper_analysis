# F. Comparison with Other Weight-only Quantization Methods

In this section, we compare SqueezeLLM with more recent weight-only quantization methods including QuIP [\(Chee et al.,](#page-9-19) [2024\)](#page-9-19) and OmniQuant [\(Shao et al.,](#page-10-23) [2023\)](#page-10-23).

## F.1. Comparison with QuIP

Here, we provide a quantitative comparison of our method to QUIP. Given that the QuIP paper only reports performance evaluation of LLaMA2-70B among all LLaMA models, we enrich our comparison by additionally incorporating our own reproduction based on their official codebase. Different from other experiments that use sequence length of 2048, we use sequence length of 4096, following the perplexity evaluation method of the QuIP paper. In Tab. [F.8,](#page-18-1) we compare the perplexity scores on Wikitext2 for LLaMA2 13B and 70B models quantized to 4, 3, and 2-bit. Note that we did not include a comparison on LLaMA2 7B as we were unable to achieve reasonable performance with QuIP, as was also reported in [\(Egiazarian et al.,](#page-9-20) [2024\)](#page-9-20).

The table indicates that dense-only SqueezeLLM consistently achieves superior performance over QUIP, across all model sizes and quantization bitwidth. With 2bit quantization, we noticed that solely relying on dense-only quantization may not yield results as competitive as those of QuIP. However, by incorporating just 0.1% sparsity (additional 0.05 bit; 0.05% outlier values + 0.05% sensitive values), SqueezeLLM significantly outperforms QuIP by a considerable margin.

#### F.2. Comparison with OmniQuant

In Tab. [F.9,](#page-19-1) we compare the perplexity of our method to OmniQuant on WikiText2 using sequence length of 2048. In particular, the table reports the perplexity numbers of 4 and 3-bit quantized models across all LLaMA and LLaMA2 models.

<span id="page-19-1"></span>Table F.9. Perplexity on Wikitext2 of all LLaMA and LLaMA2 models quantized into 4 and 3 bits using SqueezeLLM and Omni-Quant [\(Chee et al.,](#page-9-19) [2024\)](#page-9-19). For OmniQuant, we directly use the perplexity numbers that are reported in the original paper.

| Model      | Config.       | Avg. Bit Width | 7B   | 13B  | 30B  | 65B  | 2-7B | 2-13B | 2-70B |
|------------|---------------|----------------|------|------|------|------|------|-------|-------|
| Baseline   | 16-bit        | 16             | 5.68 | 5.09 | 4.1  | 3.53 | 5.47 | 4.88  | 3.32  |
| Omniquant  | 4-bit         | 4              | 5.86 | 5.21 | 4.25 | 3.71 | 5.74 | 5.02  | 3.47  |
| SqueezeLLM | 4-bit         | 4.05           | 5.79 | 5.18 | 4.22 | 3.76 | 5.62 | 4.99  | 3.41  |
| Omniquant  | 4-bit (g128)  | 4.24           | 5.77 | 5.17 | 4.19 | 3.62 | 5.58 | 4.95  | 3.4   |
| SqueezeLLM | 4-bit (0.45%) | 4.27           | 5.77 | 5.17 | 4.18 | 3.63 | 5.57 | 4.96  | 3.39  |
| Omniquant  | 3-bit         | 3              | 6.49 | 5.68 | 4.74 | 4.04 | 6.58 | 5.58  | 3.92  |
| SqueezeLLM | 3-bit         | 3.02           | 6.32 | 5.60 | 4.66 | 4.05 | 6.18 | 5.36  | 3.77  |
| Omniquant  | 3-bit (g128)  | 3.24           | 6.15 | 5.44 | 4.56 | 3.94 | 6.03 | 5.28  | 3.78  |
| SqueezeLLM | 3-bit (0.45%) | 3.24           | 6.13 | 5.45 | 4.44 | 3.88 | 5.96 | 5.23  | 3.63  |

<span id="page-19-2"></span>Table F.10. Perplexity on Wikitext2 of all LLaMA2 models quantized into 2 bits using SqueezeLLM and OmniQuant [\(Chee et al.,](#page-9-19) [2024\)](#page-9-19). For OmniQuant, we directly use the perplexity numbers that are reported in the original paper.

| Model      | Config.       | Avg. Bit Width | 2-7B  | 2-13B | 2-70B |
|------------|---------------|----------------|-------|-------|-------|
| Baseline   | 16-bit        | 16             | 5.47  | 4.88  | 3.32  |
| OmniQuant  | 2-bit         | 2              | 37.37 | 17.21 | 7.81  |
| SqueezeLLM | 2-bit         | 2.01           | 35.49 | 41.02 | 9.44  |
| SqueezeLLM | 2-bit (0.1%)  | 2.05           | 13.64 | 8.56  | 5.38  |
| OmniQuant  | 2-bit (g128)  | 2.24           | 11.06 | 8.26  | 6.55  |
| SqueezeLLM | 2-bit (0.45%) | 2.22           | 10.79 | 7.91  | 4.99  |

<span id="page-19-0"></span>Table G.11. Latency (s) and peak memory usage (GB) of 3-bit LLaMA when generating 1024 tokens on an A6000 GPU. The table compares the FP16 baseline, non-grouped and grouped GPTQ with activation ordering, and SqueezeLLM with different sparsity levels. For comparison, we include bitwidth and perplexity on the C4 benchmark.

| Method             | Bit<br>width |      | 7B    | PPL (C4) Lat (s) Mem (G) | PPL (C4) Lat (s) Mem (G) | 13B   |      |      | 30B   | PPL (C4) Lat (s) Mem (G) |      | 65B   | PPL (C4) Lat (s) Mem (G) |
|--------------------|--------------|------|-------|--------------------------|--------------------------|-------|------|------|-------|--------------------------|------|-------|--------------------------|
| Baseline           | 16           | 7.08 | 26.5  | 13.1                     | 6.61                     | 47.0  | 25.2 | 5.98 | OOM   | OOM                      | 5.62 | OOM   | OOM                      |
| GPTQ               | 3            | 7.55 | 12.6  | 3.3                      | 6.22                     | 19.1  | 6.0  | 5.76 | 36.8  | 13.8                     | 5.58 | 60.2  | 26.2                     |
| SqueezeLLM         | 3.02         | 6.32 | 13.6  | 3.4                      | 5.60                     | 21.2  | 6.1  | 4.66 | 37.8  | 16.1                     | 4.05 | 66.9  | 29.9                     |
| GPTQ (g128)        | 3.25         | 6.27 | 110.7 | 3.4                      | 5.47                     | 176.1 | 6.2  | 4.83 | 500.8 | 14.3                     | 4.55 | 955.2 | 27.3                     |
| SqueezeLLM (0.45%) | 3.24         | 6.13 | 14.6  | 3.6                      | 5.45                     | 22.2  | 6.5  | 4.44 | 42.5  | 17.4                     | 3.88 | 82.35 | 32.4                     |

For OmniQuant, we directly use the numbers reported in the original paper. Omniquant and SqueezeLLM are grouped in the table so that their model sizes are roughly the same. This comparison demonstrates that SqueezeLLM generally outperforms OmniQuant with the same model size and memory constraints.

Additionally, Tab. [F.10](#page-19-2) demonstrates the same comparison using 2-bit quantization. With 2-bit quantization, the table shows that OmniQuant without grouping outperforms dense-only SqueezeLLM on the 13B and 70B models. This can be attributed to OmniQuant's learnable clipping ranges via a few iterations of training that effectively account for outliers. SqueezeLLM's sensitivity-based nonuniform quantization alone does not inherently address this. Handling outliers can be particularly critical for 2-bit quantization where weights should be represented with only four values. Nevertheless, introducing a 0.1% sparsity remarkably enhances SqueezeLLM's performance with a minimal memory overhead increase of 0.05 bit. This perplexity improvement is also persistent when comparing OmniQuant with a group size 128 and SqueezeLLM at a 0.45% sparsity level with roughly the same size.


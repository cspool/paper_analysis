# <span id="page-7-0"></span>7 Conclusions

In this paper, we analyzed block-wise absmax quantization for large language models (LLMs) and derived an expectation-maximization algorithm to minimize the quantization error. The resulting

<span id="page-8-0"></span>Table 2: Inference results of 4-bit scalar quantization methods evaluated using multiple LLMs with block size I = 64. The evaluated metrics are the perplexity on the WikiText-2 and LAMBADA dataset, and the accuracy on the MMLU (few-shot), ARC-Challenge, HellaSwag, PIQA, SIQA, and WinoGrande benchmarks. Best result in each column in bold, second best underlined, BF16 excluded.

| Model     | Quantizer    | WikiText2<br>PPL ↓ | Lambada<br>PPL ↓ | MMLU<br>ACC ↑ | ARC-C<br>ACC ↑ | HellaSwag<br>ACC ↑ | PIQA<br>ACC ↑ | SIQA<br>ACC ↑ | WinoGrande<br>ACC ↑ | NAV<br>ACC ↑ |
|-----------|--------------|--------------------|------------------|---------------|----------------|--------------------|---------------|---------------|---------------------|--------------|
|           | BF16         | 10.12              | 4.90             | 54.0          | 42.4           | 55.3               | 76.7          | 47.2          | 69.0                | 35.7         |
|           | NF4          | 10.72              | 5.45             | 52.3          | 41.0           | 54.4               | 76.3          | 47.0          | 68.3                | 34.4         |
| Llama-3.2 | AF4          | 10.74              | 5.51             | 52.8          | 40.5           | 54.4               | 76.6          | 47.4          | 69.3                | 35.0         |
| 3B        | BOF4 (MSE)   | 10.73              | 5.35             | 52.6          | 42.1           | 54.0               | 76.7          | 46.4          | 68.8                | 34.8         |
|           | + OPQ        | 10.67              | 5.17             | 52.9          | 42.1           | 54.1               | 76.9          | 46.4          | 68.4                | 34.8         |
|           | BOF4-S (MSE) | 10.67              | 5.32             | 52.6          | 42.0           | 54.3               | 76.1          | 46.3          | 68.5                | 34.5         |
|           | + OPQ        | 10.64              | 5.25             | 52.5          | 41.8           | 54.2               | 76.2          | 46.5          | 69.5                | 34.9         |
|           | BF16         | 12.42              | 5.91             | 65.1          | 44.6           | 55.0               | 78.1          | 49.6          | 68.5                | 39.5         |
|           | NF4          | 12.36              | 7.16             | 63.0          | 43.1           | 53.6               | 77.7          | 50.8          | 67.2                | 38.2         |
| Qwen-2.5  | AF4          | 13.08              | 6.82             | 63.3          | 43.5           | 54.2               | 78.1          | 50.6          | 68.5                | 38.9         |
| 3B        | BOF4 (MSE)   | 12.46              | 6.84             | 63.5          | 46.2           | 53.8               | 77.7          | 49.8          | 68.5                | 39.2         |
|           | + OPQ        | 12.48              | 6.90             | 63.1          | 46.2           | 54.1               | 77.5          | 50.2          | 67.6                | 38.9         |
|           | BOF4-S (MSE) | 12.50              | 6.53             | 63.5          | 46.5           | 53.8               | 77.3          | 50.0          | 68.2                | 39.1         |
|           | + OPQ        | 12.35              | 6.43             | 63.5          | 46.6           | 54.0               | 77.5          | 50.8          | 69.1                | 39.7         |

<span id="page-8-1"></span>Table 3: Prompt-level and instruction-level accuracy (%) on the IFEval benchmark after finetuning Llama-3.2 3B for instruction following using 4-bit quantization with block size I = 64.

<span id="page-8-2"></span>Table 4: Accuracy (%) on the HumanEval+ and MBPP+ benchmarks after fine-tuning Llama-3.2 3B for code generation using 4-bit quantization with block size I = 64.

| Prompt-level<br>ACC ↑ | Instrlevel<br>ACC ↑                          | AVG<br>ACC ↑                                 |
|-----------------------|----------------------------------------------|----------------------------------------------|
| 21.1                  | 33.6                                         | 27.3                                         |
|                       |                                              | 28.8                                         |
|                       |                                              | 29.7                                         |
|                       |                                              | 28.7                                         |
|                       |                                              | 31.6                                         |
|                       |                                              | 29.9                                         |
|                       |                                              | 29.8                                         |
| 25.0                  | 35.0                                         | 30.0                                         |
|                       | 23.5<br>24.4<br>23.3<br>26.8<br>25.0<br>24.4 | 34.2<br>35.0<br>34.1<br>36.5<br>34.8<br>35.1 |

|              | MBPP+<br>ACC ↑ | HumanEval+<br>ACC ↑ | AVG<br>ACC ↑ |
|--------------|----------------|---------------------|--------------|
| Base Model   | 34.9           | 17.1                | 26.0         |
| BF16         | 37.8           | 30.5                | 34.2         |
| NF4          | 34.1           | 24.4                | 29.3         |
| AF4          | 32.8           | 23.2                | 28.0         |
| BOF4 (MSE)   | 34.4           | 24.4                | 29.4         |
| +OPQ         | 35.4           | 24.4                | 29.9         |
| BOF4-S (MSE) | 35.7           | 26.2                | 31.0         |
| + OPQ        | 36.5           | 27.4                | 32.0         |

family of quantizers, termed 4-bit block-wise optimal float (BOF4), reduces the weight quantization error over previously published block-wise absmax quantizers such as NF4 [\[5\]](#page-9-2) and AF4 [\[7\]](#page-9-4). We also presented an improvement to the normalization technique by normalizing blocks of weights using their *signed* absolute maximum rather than the absolute maximum, which further reduces the quantization error and empirically mitigates the negative effect of quantization on perplexity. Our experimental study confirmed the importance of precisely representing zero and outlier network weights, and found that optimization w.r.t. the mean squared error (MSE) criterion results in lower perplexity compared to mean absolute error (MAE) optimization. Finally, we introduced outlier-preserving quantization (OPQ), a mixed-precision strategy for block-wise absmax quantization, which yields a significant perplexity advantage, especially at larger block sizes. We find that our methods can outperform NF4 and AF4 not only for inference, but also when used for fine-tuning with quantization [\[5\]](#page-9-2), achieving higher accuracy on the target tasks.

Overall, our proposed methods can enable improved fine-tuning and inference for LLMs on consumergrade hardware by boosting performance without increasing the memory footprint, thereby facilitating broader participation in both the scientific investigation and the application of LLMs.


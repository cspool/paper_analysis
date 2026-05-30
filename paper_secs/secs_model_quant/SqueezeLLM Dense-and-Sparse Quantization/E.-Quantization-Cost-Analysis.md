# E. Quantization Cost Analysis

#### E.1. Memory Requirement

In Tab. [E.5,](#page-17-1) we report the memory requirement of SqueezeLLM when quantizing different model sizes from 7B to 65B. Note that our method can have a higher memory requirement than GPTQ. This is because SqueezeLLM performs quantization based on minimizing the perturbation to the loss function of the model which requires computing the Fisher information matrix. GPTQ, on the other hand, performs quantization by minimizing the perturbation to the output activation of the individual layer, which does not require back-propagating the gradient through the model to compute the Fisher information matrix. However, this is a one-time cost, and as demonstrated below, this gradient computation process is fast, taking only 2-3 minutes even for the largest 65B model.

## E.2. Quantization Time

In Tab. [E.6,](#page-17-2) we additionally assess the end-to-end time for (i) computing the Fisher information on an A100 system and (ii) performing sensitivity-based K-means clustering on Intel Xeon Gold 6126 with 48 cores, which are two major procedures in SqueezeLLM. Note that the time for computing the Fisher information matrix is minimal, taking only 2.5 minutes with the largest 65B model. K-mean clustering can take 11 min for the 7B model and up to 80 min for the 65B model. Overall, the computational time requirement of SqueezeLLM is on par with that of GPTQ.

#### E.3. Data Efficiency

In Tab. [E.7,](#page-18-0) we provide data efficiency analysis in terms of the number of data samples to calculate the Fisher information matrix (gradients) for sensitivity-based non-uniform quantization. While we used a calibration set of 100 data samples

<span id="page-18-0"></span>Table E.7. Perplexity on C4 and Wikitext2 of the LLaMA2 7B model after 4-bit quantization, with varying sizes of the calibration dataset used for computing the Fisher information matrix.

| # Data Examples | C4   | Wikitext2 |
|-----------------|------|-----------|
| 1               | 7.89 | 6.41      |
| 2               | 7.81 | 6.22      |
| 5               | 7.73 | 6.20      |
| 10              | 7.72 | 6.17      |
| 20              | 7.72 | 6.16      |
| 100             | 7.72 | 6.18      |

<span id="page-18-1"></span>Table F.8. Perplexity on Wikitext2 of the LLaMA2 13B and 70B models quantized into 4, 3, and 2 bits using SqueezeLLM and QuIP [\(Chee](#page-9-19) [et al.,](#page-9-19) [2024\)](#page-9-19). For QuIP, we use the perplexity numbers that are reported in the original paper as well as our own reproduction using the official codebase. Following the perplexity evaluation method of the QuIP paper, we use sequence length of 4096 (different from other experiments that use sequence length of 2048).

| Model                 | Config.       | Avg. Bit Width | LLaMA2-13B | LLaMA2-70B |
|-----------------------|---------------|----------------|------------|------------|
| QuIP (original paper) | 4-bit         | 4              | -          | 3.53       |
| QuIP (our repr)       | 4-bit         | 4              | 4.81       | 3.65       |
| SqueezeLLM            | 4-bit         | 4.05           | 4.67       | 3.21       |
| QuIP (original paper) | 3-bit         | 3              | -          | 3.85       |
| QuIP (our repr)       | 3-bit         | 3              | 5.25       | 3.84       |
| SqueezeLLM            | 3-bit         | 3.02           | 5.01       | 3.55       |
| QuIP (original paper) | 2-bit         | 2              | -          | 6.33       |
| QuIP (our repr)       | 2-bit         | 2              | 20.54      | 6.20       |
| SqueezeLLM            | 2-bit         | 2.01           | 61.25      | 10.86      |
| SqueezeLLM            | 2-bit + 0.1%  | 2.05           | 7.91       | 5.04       |
| SqueezeLLM            | 2-bit + 0.45% | 2.22           | 7.43       | 4.71       |

throughout the paper, a calibration set with as few as 10 examples is typically sufficient to achieve the desired quantization performance. Note that both GPTQ and AWQ require 100-200 data points for calibration as reported in the AWQ paper [\(Lin](#page-10-7) [et al.,](#page-10-7) [2023\)](#page-10-7).


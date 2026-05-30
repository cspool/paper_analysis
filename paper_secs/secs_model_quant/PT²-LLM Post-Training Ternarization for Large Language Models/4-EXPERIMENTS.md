# 4 EXPERIMENTS

### 4.1 EXPERIMENTAL SETTINGS

**Implementation Details.** All experiments are conducted using PyTorch (Paszke et al., 2019b) and Huggingface (Paszke et al., 2019a) on a single NVIDIA A800-80GB GPU. As PT<sup>2</sup>-LLM is a PTQ framework, it requires no training or gradient backpropagation. Following Li et al. (2025a) and Huang et al. (2025), we use 128 calibration samples from the Wikitext2 (Merity et al., 2017) dataset, each with a sequence length of 2048. All quantized models use a fixed block size of 128.

Models and Evaluation. We conduct comprehensive experiments on the LLaMA (Touvron et al., 2023a), LLaMA-2 (Touvron et al., 2023b), and LLaMA-3 families (Dubey et al., 2024), as well as the more recent Qwen3 series (Yang et al., 2025). Following prior work (Frantar et al., 2023; Lin et al., 2024b), we evaluate model performance in terms of both perplexity and accuracy. We report perplexity on WikiText2 (Merity et al., 2017) and C4 (Raffel et al., 2020) using a sequence length of 2048 tokens, and assess zero-shot accuracy on seven widely-used QA benchmarks: ARC-c (Clark et al., 2018), ARC-e (Clark et al., 2018), BoolQ (Clark et al., 2019), HellaSwag (Zellers et al., 2019), OBQA (Mihaylov et al., 2018), PIQA (Bisk et al., 2020), and Winogrande (Sakaguchi et al., 2020).

<span id="page-7-0"></span>Table 2: Ablation studies conducted on LLaMA-2-7B and LLaMA-3-8B. We report perplexity on Wikitext2 and C4, as well as average accuracy across seven zero-shot tasks.

(a) Effectiveness of ITF and AGA

(b) Effectiveness of SSR

| Model       | ITF | AGA | Wikitext2 ↓ | <b>C4</b> ↓ | Avg. Acc↑ |
|-------------|-----|-----|-------------|-------------|-----------|
|             | X   | X   | 22.88       | 222.17      | 37.11     |
| LLaMA-2-7B  | 1   | X   | 15.47       | 34.17       | 38.12     |
| LLaWIA-2-/D | X   | 1   | 12.25       | 26.17       | 42.86     |
|             | 1   | 1   | 11.56       | 24.38       | 43.33     |
|             | X   | Х   | 247.75      | 1227.94     | 33.26     |
| LLaMA-3-8B  | 1   | X   | 83.76       | 1039.80     | 33.90     |
| LLaWIA-3-6D | X   | 1   | 47.83       | 520.14      | 35.29     |
|             | 1   | 1   | 32.19       | 129.83      | 37.79     |

| Model      | Reorder | Method        | Wikitext2 $\downarrow$ | <b>C4</b> ↓ | Avg. Acc ↑ |
|------------|---------|---------------|------------------------|-------------|------------|
|            | X       | -             | 13.06                  | 27.66       | 41.37      |
| LLaMA-2-7B | /       | Random        | 12.84                  | 28.24       | 40.86      |
| LLaWA-2-7B | ✓       | Hessian-based | 12.35                  | 25.44       | 39.15      |
|            | ✓       | SSR           | 11.56                  | 24.38       | 43.33      |
|            | X       | -             | 112.83                 | 599.19      | 33.08      |
| LLaMA-3-8B | ✓       | Random        | 113.42                 | 466.07      | 33.37      |
| LLaWA-3-0D | ✓       | Hessian-based | 35.86                  | 131.33      | 37.31      |
|            | ✓       | SSR           | 32.19                  | 129.83      | 37.79      |

(c) Ablation study on calibration set size

(d) Ablation study on calibration set type

| Model      | Calib. Set Size | Wikitext2 ↓ | <b>C4</b> ↓ | Avg. Acc↑ |
|------------|-----------------|-------------|-------------|-----------|
|            | 64              | 11.92       | 25.27       | 43.31     |
| LLaMA-2-7B | 128             | 11.56       | 24.38       | 43.33     |
|            | 256             | 11.35       | 23.52       | 43.55     |
|            | 64              | 38.90       | 252.19      | 35.20     |
| LLaMA-3-8B | 128             | 32.19       | 129.83      | 37.79     |
|            | 256             | 32.25       | 167.48      | 37.95     |

| Model      | Calib. Data Type | Wikitext2 ↓ | <b>C4</b> ↓ | Avg. Acc↑ |
|------------|------------------|-------------|-------------|-----------|
|            | Wikitext2        | 11.56       | 24.38       | 43.33     |
| LLaMA-2-7B | C4               | 18.94       | 20.15       | 43.32     |
|            | PTB              | 27.52       | 35.15       | 41.15     |
|            | Wikitext2        | 32.19       | 129.83      | 37.79     |
| LLaMA-3-8B | C4               | 168.81      | 72.00       | 37.80     |
|            | РТВ              | 428.97      | 579.57      | 35.21     |

**Baselines.** We compare PT<sup>2</sup>-LLM against a diverse set of representative PTQ methods operating in the 2-bit and sub-2-bit regimes. Slim-LLM (Huang et al., 2025) serves as a strong baseline for mixed-precision quantization, achieving high performance with an average of 2 bits. PB-LLM (Shang et al., 2024) targets the sub-2-bit regime and is closest to ours in average bitwidth, making it a relevant baseline. We further include GPTQ (Frantar et al., 2023) and AWQ (Lin et al., 2024b) as widely used baselines, along with QuIP (Chee et al., 2025), which targets 2-bit quantization.

#### 4.2 Comparisons with State-of-the-Art Methods

Table 1 summarizes the results of PT<sup>2</sup>-LLM and baselines on LLaMA, LLaMA-2, LLaMA-3, and Qwen3-base reporting WikiText2/C4 perplexity, zero-shot accuracy on seven tasks (with average), and average bitwidth. Despite operating at the lowest bitwidth (1.58), PT<sup>2</sup>-LLM consistently ranks among the top two in both perplexity and average accuracy across all model sizes. It clearly outperforms 2-bit baselines such as GPTQ, AWQ, and QuIP. Compared to Slim-LLM, the current SOTA 2-bit method, PT<sup>2</sup>-LLM achieves higher average accuracy on all models except LLaMA-2-7B, where it performs comparably. Relative to PB-LLM with comparable bitwidth, PT<sup>2</sup>-LLM delivers significant gains: on LLaMA-7B, it improves average accuracy from 33.44 to 45.07, reduces WikiText2 perplexity by 86%, and requires less memory. Additional results are provided in the supplementary file.


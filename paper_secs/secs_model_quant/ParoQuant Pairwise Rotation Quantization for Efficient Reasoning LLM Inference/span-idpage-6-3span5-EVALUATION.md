# <span id="page-6-3"></span>5 EVALUATION

Models and Tasks. We apply ParoQuant on LLaMA-2 (7B) [\(Touvron et al.,](#page-11-10) [2023\)](#page-11-10), LLaMA-3 (8B, 70B) & LLaMA-3.1 Instruct (8B) [\(Grattafiori et al.,](#page-9-5) [2024\)](#page-9-5), DeepSeek-R1-distilled LLaMA-3.1 (8B) [\(Guo et al.,](#page-9-4) [2025\)](#page-9-4), and Qwen3 (1.7B, 4B, 8B, 14B) [\(Yang et al.,](#page-11-4) [2025\)](#page-11-4) pre-trained models. We evaluate the quantization quality with three types of evaluation: (1) *Perplexity* on WikiText2 [\(Merity](#page-10-10) [et al.,](#page-10-10) [2017\)](#page-10-10) and C4 [\(Dodge et al.,](#page-9-9) [2021\)](#page-9-9); (2) *Reasoning accuracy* on MMLU-Pro [\(Wang et al.,](#page-11-5) [2024\)](#page-11-5), GPQA Diamond [\(Rein et al.,](#page-10-11) [2024\)](#page-10-11), AIME-24, and AIME-25 [\(MAA,](#page-10-12) [2025\)](#page-10-12); (3) *Non-reasoning accuracy* on BoolQ [\(Clark et al.,](#page-9-10) [2019\)](#page-9-10), ARC-Challenge, ARC-Easy [\(Clark et al.,](#page-9-11) [2018\)](#page-9-11), and HellaSwag [\(Zellers et al.,](#page-11-11) [2019\)](#page-11-11).

Implementation. We focus on 4-bit weight-only linear quantization with a group size of 128. Linear quantization is more efficient and is widely supported by existing frameworks. The choice of 4 bits and a 128 group size offers the optimal trade-off between accuracy and bit width for linear quantization [\(Dettmers & Zettlemoyer,](#page-9-6) [2023\)](#page-9-6). We apply 8 independent rotations on each 128-channel group, with each rotation consisting of up to 64 pairs. Each layer is optimized for 10 epochs at each stage using AdamW [\(Loshchilov & Hutter,](#page-10-13) [2019\)](#page-10-13) with a fixed set of hyperparameters for all experiments, except for the 70B model, where we adjust the batch size to accommodate memory constraints. To reduce the risk of overfitting to one dataset, we use a training set of 2048 samples drawn evenly from WikiText2, C4, and RedPajama [\(Weber et al.,](#page-11-12) [2024\)](#page-11-12), and select the best parameters using 64 validation samples from Pile [\(Gao et al.,](#page-9-12) [2020\)](#page-9-12). More details are provided in Section [A.4.](#page-15-0)

Baselines. We compare the accuracy and efficiency of ParoQuant with three weight-only PTQ baselines. AWQ [\(Lin et al.,](#page-10-0) [2024b\)](#page-10-0) optimizes channel-wise scaling with grid search and is the most used 4-bit weight-only quantization method. EfficientQAT [\(Chen et al.,](#page-9-3) [2025\)](#page-9-3) achieves stateof-the-art linear quantization accuracy with layer-wise fine-tuning of weights and quantization parameters[\\*](#page-6-1). QTIP [\(Tseng et al.,](#page-11-2) [2024b\)](#page-11-2) is the state-of-the-art vector quantization method utilizing randomized Hadamard transform and an advanced trellis quantization algorithm. In addition, we include the perplexity results of QuIP# [\(Tseng et al.,](#page-11-8) [2024a\)](#page-11-8), a vector-quantization predecessor of QTIP that also adopts the Hadamard transform, and two weight-activation linear quantization methods, OmniQuant [\(Shao et al.,](#page-10-1) [2024\)](#page-10-1) and SpinQuant [\(Liu et al.,](#page-10-3) [2025b\)](#page-10-3), which are also applicable for weight-only quantization. We apply block-wise quantization with a group size of 128 on all linear quantization methods and the corresponding default settings on vector quantization methods.

### 5.1 ACCURACY RESULTS

Perplexity. Table [1](#page-7-0) shows the perplexity results of 4-bit quantized models ranging in size from 1.7B to 70B. Among linear quantization methods, ParoQuant achieves state-of-the-art quantization

<span id="page-6-1"></span><sup>\*</sup>We only apply the "Block-AP" stage of EfficientQAT, as its "E2E-QP" stage involves supervised fine-tuning, which is out of the scope of PTQ.

<span id="page-7-2"></span>accuracy across all sizes, particularly on challenging cases like LLaMA-3 and smaller models under 4B. It also delivers strong performance compared with rotation-based methods including QuIP#, QTIP, and SpinQuant. It outperforms QuIP# and matches QTIP on all models, despite the inherently larger error of linear quantization, highlighting the superior effectiveness of our proposed transform over the Hadamard transform (see Section A.2 for detailed analysis). Moreover, ParoQuant provides a decent speedup over these two methods. This underscores the efficiency of our proposed transform.

<span id="page-7-0"></span>

| M-41 1         | Т      |      | WikiText2<br>L3-8 L3-70 L2-7 Q3-1.7 Q3-4 Q3-8 Q3-14 |      |           |      |      |       | C4   |       |      |        |      |      |           |                                               |
|----------------|--------|------|-----------------------------------------------------|------|-----------|------|------|-------|------|-------|------|--------|------|------|-----------|-----------------------------------------------|
| wichiou        | Type   | L3-8 | L3-70                                               | L2-7 | Q3-1.7    | Q3-4 | Q3-8 | Q3-14 | L3-8 | L3-70 | L2-7 | Q3-1.7 | Q3-4 | Q3-8 | Q3-14     | Speedup                                       |
| FP16           | -      | 5.54 | 2.56                                                | 5.12 | 8.32      | 7.01 | 6.24 | 5.70  | 7.10 | 5.78  | 6.63 | 8.62   | 7.61 | 6.97 | 6.54      | 1.0×                                          |
| QUIP#<br>QTIP  |        |      |                                                     |      | -<br>8.46 |      |      |       |      |       |      |        |      |      | -<br>6.57 | 1.9×<br>1.7×                                  |
| AWQ<br>OmniQ   |        |      |                                                     |      |           |      |      |       |      |       |      |        |      |      | 6.65<br>- | 2.4×<br>2.4× <sup>†</sup>                     |
| SPINQ<br>E-QAT | linear | 5.87 | 3.33                                                | 5.22 | 8.60      | 7.19 | 6.37 | 5.82  | 7.36 | 6.72  | 6.76 |        | 7.77 | 7.08 | 6.63      | $2.4 \times^{\dagger}$ $2.4 \times^{\dagger}$ |
| PAROQ          | linear | 5.73 | 2.82                                                | 5.17 | 8.44      | 7.10 | 6.29 | 5.75  | 7.27 | 5.86  | 6.73 | 8.74   | 7.70 | 7.04 | 6.59      | $2.2 \times$                                  |

 $<sup>\</sup>dagger$  Uses results of AWQ as a reference, as the method does not incur significant overhead from the transform.

Table 1: Perplexity (↓) results of 4-bit models. The context length is 8192 for LLaMA-3 and Qwen3 (base models), and 4096 for LLaMA-2. The best results among linear quantization methods are in **bold**. Speedup over FP16 models is reported as the geometric mean across Q3-1.7, Q3-4, L3-8, Q3-14, measured on an RTX A6000 with a batch size of 1 during decoding.

**Reasoning Tasks.** Table 2 shows the accuracy results of four reasoning benchmarks: MMLU-Pro (12k samples), GPQA Diamond (198 samples), AIME-24 (30 samples), and AIME-25 (30 samples). On the larger MMLU-Pro benchmark, ParoQuant consistently outperforms all linear quantization baselines and matches the accuracy of QTIP. While results on the smaller GPQA and AIME benchmarks exhibit more randomness due to the limited number of samples, ParoQuant still outperforms the baselines in most cases. Overall, ParoQuant causes only an average **0.9**% accuracy degradation and achieves **6.3**%, **2.4**%, and **0.9**% improvements over EfficientQAT, AWQ, and QTIP, respectively. This demonstrates ParoQuant's superior quantization accuracy in long generation.

<span id="page-7-1"></span>

|        |        | R1-Distill-Llama-8B |      |            |            | Qwen3-4B |      |            |            | Qwen3-8B |      |             | Qwen3-14B  |      |      |            |            |      |
|--------|--------|---------------------|------|------------|------------|----------|------|------------|------------|----------|------|-------------|------------|------|------|------------|------------|------|
| Method | Туре   | MMLU                | GPQA | AIME<br>24 | AIME<br>25 | MMLU     | GPQA | AIME<br>24 | AIME<br>25 | MMLU     | GPQA | AIME<br>24  | AIME<br>25 | MMLU | GPQA | AIME<br>24 | AIME<br>25 | Avg. |
| FP16   | _      | 58.8                | 46.6 | 42.2       | 32.2       | 71.0     | 50.0 | 75.6       | 62.2       | 74.6     | 60.3 | 75.6        | 72.2       | 78.1 | 62.5 | 73.3       | 68.9       | 62.8 |
| QTIP   | vector | 57.4                | 43.4 | 37.8       | 30.1       | 69.7     | 55.2 | 67.8       | 58.9       | 74.0     | 59.3 | 72.2        | 63.3       | 77.9 | 64.0 | 76.7       | 69.0       | 61.0 |
| AWQ    |        |                     |      |            |            |          |      |            |            |          |      |             |            |      |      |            |            |      |
| E-QAT  |        |                     |      |            |            |          |      |            |            |          |      |             |            |      |      |            |            |      |
| PAROQ  | linear | 57.1                | 47.5 | 36.6       | 31.1       | 70.1     | 53.7 | 73.3       | 63.3       | 74.1     | 57.7 | <b>75.6</b> | 63.3       | 77.5 | 63.5 | 77.8       | 67.8       | 61.9 |

Table 2: Zero-shot accuracy (↑) on reasoning tasks. Best linear quantization results are in **bold**.

**Non-Reasoning Tasks.** Table 3 shows the zero-shot accuracy on commonsense benchmarks with thinking mode disabled. ParoQuant maintains near-lossless performance, outperforming AWQ, EfficientQAT, and QTIP by **0.9%**, **0.7%**, and **0.2%**, respectively. The accuracy gap is smaller than in reasoning tasks because these benchmarks evaluate only a few generated tokens, so error accumulation is minimal.

#### 5.2 EFFICIENCY RESULTS

Table 4 shows the decoding throughput on an RTX A6000. To ensure a fair comparison, we implement all methods on top of the Transformers library (Wolf et al., 2020), modifying only the weight transform and dequantization code (details and more results are in Section A.5). ParoQuant is only about 10% slower than AWQ while providing a significant accuracy improvement, and it matches the accuracy of QTIP while being 15%-30% faster. For the training efficiency, see Section A.6 for more details.

<span id="page-8-1"></span>

|                                                         |                  | LLaMA-3.1-8B-Instruct |                                                                                                 |              |                     | Qwen3-4B                         |  |              |                     | Qwen3-8B                         |  |              | Qwen3-14B           |                                  |  |              |                |                        |
|---------------------------------------------------------|------------------|-----------------------|-------------------------------------------------------------------------------------------------|--------------|---------------------|----------------------------------|--|--------------|---------------------|----------------------------------|--|--------------|---------------------|----------------------------------|--|--------------|----------------|------------------------|
| Method Type                                             |                  |                       | BoolQ ARC-C ARC-E HSwag BoolQ ARC-C ARC-E HSwag BoolQ ARC-C ARC-E HSwag BoolQ ARC-C ARC-E HSwag |              |                     |                                  |  |              |                     |                                  |  |              |                     |                                  |  |              |                | Avg.                   |
| FP16                                                    | –                |                       | 84.1 51.7                                                                                       | 81.8         |                     | 59.1 85.1 50.8                   |  | 80.5         |                     | 52.3 86.6 55.8                   |  | 83.5         |                     | 57.1 89.4 58.6                   |  | 84.2         |                | 60.9 70.1              |
| QTIP                                                    | vector 84.3 51.8 |                       |                                                                                                 | 81.6         |                     | 58.9 85.0 50.0                   |  | 79.8         |                     | 51.8 86.9 54.9                   |  | 82.8         |                     | 57.0 89.2 57.6                   |  | 83.5         |                | 60.8 69.7              |
| AWQ<br>E-QAT linear 83.5 51.9<br>PAROQ linear 83.9 52.1 | linear 83.5 51.7 |                       |                                                                                                 | 80.6<br>80.9 | 82.2 58.7 85.3 49.7 | 58.4 85.0 47.4<br>58.4 84.5 48.3 |  | 77.9<br>79.7 | 80.7 51.8 87.0 55.3 | 51.3 86.2 53.8<br>51.1 86.1 53.6 |  | 82.2<br>81.7 | 83.3 56.8 89.1 57.2 | 56.2 89.1 57.9<br>56.1 89.0 58.5 |  | 83.2<br>84.0 | 84.3 60.7 69.9 | 60.3 69.0<br>60.4 69.2 |

Table 3: Zero-shot accuracy (↑) on non-reasoning tasks. Best linear quantization results are in bold.

<span id="page-8-2"></span>

|                      | Qwen3-1.7B                                      |                      | Qwen3-4B                                        |                      | LLaMA-3-8B                                     |                      | Qwen3-14B                                    |                      |  |
|----------------------|-------------------------------------------------|----------------------|-------------------------------------------------|----------------------|------------------------------------------------|----------------------|----------------------------------------------|----------------------|--|
| Method               | Throughput                                      | W2 PPL               | Throughput                                      |                      | Throughput                                     | W2 PPL               | Throughput                                   | W2 PPL               |  |
| FP16                 | (1.0×)<br>170                                   | 8.32                 | (1.0×)<br>78                                    | 7.01                 | (1.0×)<br>45                                   | 5.54                 | (1.0×)<br>25                                 | 5.70                 |  |
| AWQ<br>QTIP<br>PAROQ | (1.9×)<br>320<br>(1.2×)<br>209<br>(1.6×)<br>278 | 8.80<br>8.46<br>8.44 | (2.3×)<br>176<br>(1.5×)<br>117<br>(2.1×)<br>160 | 7.36<br>7.09<br>7.10 | (2.7×)<br>120<br>(2.1×)<br>95<br>(2.5×)<br>112 | 5.92<br>5.69<br>5.73 | (2.8×)<br>70<br>(2.2×)<br>55<br>(2.6×)<br>65 | 5.85<br>5.75<br>5.75 |  |

Table 4: Decoding (with batch size of 1) throughput (tokens/s).

### <span id="page-8-0"></span>5.3 ABLATION STUDY

Table [5](#page-8-3) shows the effectiveness of each component of ParoQuant. The effects of channel-wise scaling and independent rotations are distinct, and combining both of them yields better quantization accuracy than applying either one alone. Fine-tuning weights and quantization parameters in the second optimization stage further improves the accuracy compared with directly applying RTN. For a more detailed comparison of the transforms, see Section [A.2.](#page-12-0)

Table [6](#page-8-3) shows the effects of the calibration set, calibration size, and number of independent rotations on end-to-end quantization accuracy. ParoQuant achieves surprisingly strong performance with as few as 128 training samples. Moreover, accuracy improves as the number of rotations increases up to 8, indicating improved fitting capability. We also optimize the model with 2048 calibration samples from RedPajama alone, and the results are slightly worse than with the mixed dataset. This shows that using a more diverse training set improves the generalization ability of the models.

<span id="page-8-3"></span>

|             | Transform | C4 (↓) |
|-------------|-----------|--------|
|             | None      | 7.56   |
|             | S         | 7.40   |
| w/o Stage 2 | 8 IR      | 7.50   |
|             | 8 IR + S  | 7.35   |
|             | None      | 7.42   |
|             | S         | 7.41   |
| w/ Stage 2  | 8 IR      | 7.40   |
|             | 8 IR + S  | 7.27   |

Table 5: Ablations on transforms and optimization stages with LLaMA-3-8B (S: channelwise scaling, IR: independent rotation).

| # Samples           | # IR             | C4 (↓)                       | MMLU (↑)                     |
|---------------------|------------------|------------------------------|------------------------------|
| 128                 | 8                | 7.30                         | 69.5                         |
| 512                 | 8                | 7.27                         | 69.7                         |
| 2048                | 0<br>2<br>4<br>8 | 7.41<br>7.28<br>7.27<br>7.27 | 69.6<br>69.4<br>69.4<br>70.1 |
| 2048<br>(RedPajama) | 8                | 7.27                         | 69.5                         |

Table 6: Ablations on training samples and number of rotations (IR) with LLaMA-3-8B (C4 perplexity) and Qwen3-4B (MMLU-Pro accuracy).


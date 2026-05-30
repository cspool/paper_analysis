# <span id="page-13-0"></span>A3 ABLATION STUDIES

Table A2: Effect of combination of equivalent transformation and weight clipping. We report the average perplexity of WikiText2 and C4, and the average accuracy on 6 zero-shot tasks like Table 2.

| LLaMa-7B W4A4          | Average PPL ↓ | Average Acc. ↑ |
|------------------------|---------------|----------------|
| SmoothQuant            | 28.78         | 38.41          |
| LET                    | 16.97         | 48.83          |
| LET + grid-searched WC | 15.82         | 49.59          |
| SmoothQuant + LWC      | 15.80         | 50.15          |
| LET + LWC              | 12.87         | 52.65          |

Combination of equivalent transformation and weight clipping. The synergy between LET and LWC is achieved through a sophisticated differentiable framework as demonstrated in Algorithm [1,](#page-12-3) not a simple additive combination. LET performs activation-to-weight migration, and LWC further facilitates the quantization of weights, resulting in a seamless integration of the two techniques. In Table [A2,](#page-7-0) we also test other combination variants, including replacing LET with SmoothQuant or replacing LWC with grid-searched weight clipping. The results show that training LET and LWC simultaneously achieves the best performance.

Efficacy of each component. Table [A3](#page-8-0) reveals that the baseline model incorporates both LWC and LET, labeled as 'LWC+LET'. We further investigate their contributions by removing each component. Both components positively influence performance, but LET proves essential for weightactivation quantization. Disabling it for W4A4 results in a marked increase in perplexity to e3, mainly due to challenges with activation quantization outliers. For weight-only quantization, LET significantly boosts OPT's performance but offers a slight enhancement for LLaMA, explained by Table A3: Efficacy of each component. WikiText2 perplexity1 is reported in this table. '-' indicats remove the corresponding module from the overall proposed methods.

| PPL↓       |          | LLaMA-13B | OPT-13B |       |       |
|------------|----------|-----------|---------|-------|-------|
| Method     |          | W4A4      | W3A16   | W4A4  | W3A16 |
|            | LWC+LET  | 10.87     | 5.65    | 11.65 | 10.87 |
|            | -LWC     | 20.75     | 7.65    | 15.23 | 12.98 |
| components | -LET     | 5.4e3     | 5.68    | 7.8e3 | 11.29 |
|            | -LWC-LET | 1.8e3     | 10.68   | 7.8e5 | 4.6e3 |

LLaMA's few weight outliers. For example, in naive W3A16 quantization (-LWC-LET), LLaMA reaches a perplexity of 10.68, while OPT's spikes to 4.6e3. Consequently, LET is turned off for LLaMA in weight-only quantization given its limited advantage for faster training.

<span id="page-14-0"></span>Table A4: Design choices of learnable equivalent transformation. WikiText2 perplexity1 is reported in this table.

|     | PPL↓       |       | LLaMA-13B | OPT-13B |       |  |
|-----|------------|-------|-----------|---------|-------|--|
|     | Method     | W4A4  | W3A16     | W4A4    | W3A16 |  |
|     | LWC+LET    | 10.87 | 5.65      | 11.65   | 10.87 |  |
|     | -shifting  | 11.47 | 5.65      | 13.64   | 10.87 |  |
| LET | -attention | 11.34 | 5.65      | 11.79   | 10.87 |  |

Design choices of learnable equivalent transformation. In comparison to the equivalent transformation incorporated in SmoothQuant [\(Xiao et al.](#page-11-6) [\(2023\)](#page-11-6)), our approach additionally implements channel-wise shifting and attention transformation. The effects of these innovations are evaluated in Table [A4.](#page-14-0) We can observe that both modifications enhance the performance of weight-activation quantization. However, the incremental benefit of the equivalent transformation in the attention operation is comparatively minor. This discrepancy is primarily due to the majority of outliers existing in the output of the normalization layer while being less prevalent in the Q/K/V matrix.

<span id="page-14-1"></span>Table A5: Impact of LET on each position. '-' indicates removing corresponding LET. We respectively remove the LET from each layer, and reporting the average perplexity of WikiText2 and C4, and the average accuracy on 6 zero-shot tasks like Table 2.

| LLaMa-7B                         | Average PPL ↓ | Average Acc. ↑ |
|----------------------------------|---------------|----------------|
| W4A4                             | 12.87         | 52.65          |
| -[ln1, (q proj, k proj, v proj)] | 19.87         | 46.79          |
| -[v proj, out proj]              | 13.03         | 51.68          |
| -[Q,K]                           | 13.34         | 51.47          |
| -[ln2, fc1]                      | 14.47         | 51.04          |

Impact of LET on each position. We exclude the LET of the second linear layer due to the high sparsity of features after the non-linear layer leads to unstable gradients. Therefore, we have four LET pairs, represented as [ln1, (q proj, k proj, v proj)], [v proj, out proj], [Q, K], and [ln2, fc1]. As shown in Table [A5,](#page-14-1) we can find that all four LETs can improve the performance, specially for the [ln1, (q proj, k proj, v proj)] pair. Such results also demonstrate that the activation outliers are more serious after layer normalization layers.

<span id="page-14-2"></span>Table A6: Impact of initialization of LET. We report the average perplexity of WikiText2 and C4, and the average accuracy on 6 zero-shot tasks like Table 2.

| LLaMa-7B                 | Average PPL ↓ | Average Acc. ↑ |
|--------------------------|---------------|----------------|
| W4A4                     | 12.87         | 52.65          |
| initialize scaling as 1  | 13.64         | 51.37          |
| initialize shifting as 0 | 12.95         | 52.22          |

Impact of initialization of LET. We initialize the channel-wise scaling factor with SmoothQuant [Xiao et al.](#page-11-6) [\(2023\)](#page-11-6), and initialize the channel-wise shifting with Outlier Suppression+ [Wei et al.](#page-11-9) [\(2023\)](#page-11-9). To validate the impact of careful initialization, we try to initial scaling as 1 and initial shifting as 0. As shown in Table [A6,](#page-14-2) we can find that careful initialization of scaling and shifting can improve the final performance. Specifically, scaling initialization is more important than shifting, since scaling plays the main role in alleviating outliers.

Table A7: Impact of Softmax quantization. We report the average perplexity of WikiText2 and C4, and the average accuracy on 6 zero-shot tasks like Table 2.

| LLaMa-7B             | Average PPL ↓ | Average Acc. ↑ |
|----------------------|---------------|----------------|
| W4A4 + Softmax 16bit | 12.87         | 52.65          |
| W4A4 + Softmax 8bit  | 12.91         | 51.93          |
| W4A4 + Softmax 6bit  | 13.20         | 51.70          |
| W4A4 + Softmax 4bit  | 18.80         | 48.52          |

Impact of Softmax quantization. The output of SoftMax has a long-tailed distribution, making it unsuitable for uniform quantization. We carry out experiments to quantize the Softmax output into different bit numbers. As shown in the following table, we can find that quantizing the output of softmax into 8-bit and 6-bit bring acceptable performance degeneration, which demonstrates that block-wise calibration can compensate for the loss of 8-bit and 6-bit Softmax quantization. However, 4-bit Softmax quantization brings significantly performance loss, which requires further exploration and additional trick such as log2 quantization in RepQViT [\(Li et al.,](#page-10-13) [2023\)](#page-10-13). Note that we keep the output of SoftMax as 16-bit if no special instruction.

<span id="page-15-0"></span>Table A8: Impact of iterative training of LWC and LET. We report the average perplexity of WikiText2 and C4, and the average accuracy on 6 zero-shot tasks like Table 2.

| LLaMa-7B W4A4                            | Average PPL ↓ | Average Acc. ↑ |
|------------------------------------------|---------------|----------------|
| simultaneously                           | 12.87         | 52.65          |
| each iteration                           | 13.56         | 50.91          |
| each epoch                               | 13.51         | 52.06          |
| each epoch + double training epochs 4bit | 12.80         | 52.50          |

Impact of iterative training. In our approach, LWC and LET are trained simultaneously, and we have also explored an iterative training approach by iterations or epochs. The results, as presented in Table [A8,](#page-15-0) clearly indicate that training LWC and LET simultaneously yields the best performance. This experiment demonstrates that the synergy between LET and LWC creates a progressive process, where both techniques reinforce each other rather than interfere. To further support this statement, we conducted an additional experiment (last row in Table [A8\)](#page-15-0), training LWC and LET iteratively with double training epochs. The results show that simultaneous training with 20 epochs achieves comparable performance to iterative training with 40 epochs. This demonstrates the effectiveness and efficiency of training LWC and LET simultaneously.

<span id="page-15-1"></span>Table A9: Ablation of training time. We train LLaMA-7B with different quantization configuration on 128 2048-tokens segments from WikiText2 over various epochs. '0' indicates only initialization without fine-tuning. Wikitext perplexity is reported in this table.

| Epochs | W4A16 | W3A16 | W2A16 | W6A6 | W4A4  |
|--------|-------|-------|-------|------|-------|
| 0      | 6.29  | 24.04 | 1.1e5 | 6.16 | 33.93 |
| 10     | 5.87  | 6.51  | 27.49 | 5.96 | 12.04 |
| 20     | 5.85  | 6.49  | 17.46 | 5.95 | 11.26 |
| 40     | 5.86  | 6.47  | 15.47 | 5.95 | 11.23 |
| 80     | -     | -     | 14.77 | -    | -     |
|        |       |       |       |      |       |

Training Time As illustrated in Table [A9,](#page-15-1) LLaMA-7B was trained across various epochs to determine the optimal convergence time. Most quantization configurations converge within 20 epochs, with the exception of W2A16, which necessitates 80 epochs. Consequently, we establish a training <span id="page-16-2"></span>epoch of 20 for all configurations, except for W2A16, for which we set it to 40 in consideration of the training time.

> Table A10: Ablation of calibration dataset. LLaMA-7B/PPL↓ W3A16 W4A4 Calibration Dataset WikiText2 C4 WikiText2 C4 WikiText2 6.47 8.19 11.23 14.61 C4 6.67 8.13 12.17 14.24 Pile 6.69 8.17 12.04 14.22 Varience 0.009 0.0006 0.17 0.03

Table A11: Ablation of sample number of calibration dataset.

<span id="page-16-3"></span>

| LLaMA-7B/PPL↓ | W3A16     |      | W4A4      |       |  |
|---------------|-----------|------|-----------|-------|--|
| Sample Number | WikiText2 | C4   | WikiText2 | C4    |  |
| 16            | 6.47      | 8.18 | 11.56     | 14.84 |  |
| 32            | 6.47      | 8.18 | 11.48     | 14.80 |  |
| 64            | 6.48      | 8.19 | 11.40     | 14.57 |  |
| 128           | 6.47      | 8.19 | 11.23     | 14.61 |  |
| 256           | 6.46      | 8.19 | 11.41     | 14.90 |  |

Calibration Data OmniQuant utilizes gradient optimization on constrained calibration datasets, sourced from WikiText2 and comprising 128 segments with 2048 tokens each. This prompts concerns about potential overfitting to the calibration dataset. To explore this, we evaluated the calibration dataset's influence using two other datasets: Pile [\(Gao et al.](#page-9-13) [\(2020\)](#page-9-13)) and c4 [\(Raffel et al.](#page-11-15) [\(2020\)](#page-11-15)). As depicted in Table [A10,](#page-16-2) the variance in perplexity across diverse calibration datasets is marginal, fluctuating between 0.0006 and 0.17. This underscores OmniQuant's robustness concerning calibration set distribution. Furthermore, the data efficiency of OmniQuant was gauged by modulating the number of training samples, as presented in Table [A11.](#page-16-3) Remarkably, OmniQuant converges with as few as 16 samples. Our selection of 128 samples aligns with established practices in prior works [\(Frantar et al.](#page-9-2) [\(2022\)](#page-9-2); [Lin et al.](#page-10-4) [\(2023\)](#page-10-4)).

<span id="page-16-4"></span>Table A12: Omniquant runtime on LLaMA family. The time correspond to training 128 2048-tokes segment over 20 epochs and a batch size of 1 on a single NVIDIA A100-80G.

| LLaMA             | 7B   | 13B  | 30B  | 65B   |
|-------------------|------|------|------|-------|
| weight-only       | 1.1h | 2.2h | 4.5h | 8.9h  |
| weight-activation | 1.6h | 3.3h | 7.3h | 14.4h |


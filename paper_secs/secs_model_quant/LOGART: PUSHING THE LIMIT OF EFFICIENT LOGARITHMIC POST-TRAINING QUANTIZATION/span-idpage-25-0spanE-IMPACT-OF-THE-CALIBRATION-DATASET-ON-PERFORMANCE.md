# <span id="page-25-0"></span>E IMPACT OF THE CALIBRATION DATASET ON PERFORMANCE

### E.1 IMPACT OF CALIBRATION DATASET SOURCE

The main experiments on LLMs, detailed in Table [3,](#page-7-0) utilize the C4 dataset for calibration to ensure a fair comparison with prior SOTA methods. This section evaluates the robustness of LogART by measuring its sensitivity to the calibration data source. We conduct a parallel set of experiments where models are calibrated using the WikiText-2 dataset instead of C4. All other experimental conditions, including the models and experimental settings, remain identical to those in the main text to isolate the effect of the calibration data. The results of this comparison are summarized in Table [13.](#page-25-1)

<span id="page-25-1"></span>Table 13: Comparison of performance (PPL) for per-channel 3-bit LogART on LLM weights using 32 random 2048-token segments for calibration from the C4 and WikiText-2 datasets.

| Calib. Dataset |                  | OPT-125M | OPT-1.3B | OPT-6.7B | LLaMA2-7B | LLaMA3-8B |
|----------------|------------------|----------|----------|----------|-----------|-----------|
| C4             | PPL (WikiText-2) | 31.52    | 15.53    | 11.11    | 6.31      | 8.19      |
|                | PPL (C4)         | 29.98    | 17.29    | 13.37    | 8.38      | 12.44     |
| WikiText-2     | PPL (WikiText-2) | 31.15    | 15.61    | 11.37    | 6.14      | 7.83      |
|                | PPL (C4)         | 30.44    | 17.60    | 13.54    | 8.55      | 13.27     |

Table [13](#page-25-1) reveals a distinct in-domain alignment pattern. Models generally achieve lower PPL when the calibration source matches the test domain. This indicates that the calibration set helps the quantization parameters adapt to the specific linguistic distribution of the target domain. The performance difference stems from domain alignment and outlier coverage. C4 provides high linguistic diversity, covering a wider range of activation outliers, which ensures better generalization. In contrast, WikiText-2 offers a highly consistent, formal distribution, leading to slightly lower PPL due to in-domain overfitting when used for both calibration and test.

Crucially, despite these variations, LogART exhibits strong robustness. The performance variance across different calibration sources is marginal, and the model maintains high accuracy regardless of the source. For general-purpose deployment and fair SOTA comparison, we recommend using a large-scale, diverse dataset like C4. However, for tasks targeting a specific domain, utilizing in-domain calibration data yields the best fine-grained performance.

#### E.2 IMPACT OF CALIBRATION DATASET SIZE

In addition to the choice of calibration data source, the size of the calibration dataset is also a critical factor for the practicality and efficiency of a PTQ method. A method that requires a large number of samples can be costly and time-consuming. In this section, we evaluate the sensitivity of LogART performance to the number of calibration samples. The study is conducted on the OPT-125M and LLaMA2-7B models with 3-bit per-channel weight quantization, using the calibration data from the WikiText-2 dataset. We vary the number of calibration segments, testing sizes of 32 and 128 samples (each containing 2048 tokens), and measure the resulting PPL. The results for OHS alone and for the full LogART (OHS+LLR) are presented in Table [14.](#page-25-2)

<span id="page-25-2"></span>Table 14: Comparison of performance (PPL) for different calibration dataset sizes under 3-bit per-channel weight quantization using LogART.

| Config. | Segments |             | OPT-125M |         | LLaMA2-7B   |          |          |  |
|---------|----------|-------------|----------|---------|-------------|----------|----------|--|
|         |          | PPL (WT-2)* | PPL (C4) | Runtime | PPL (WT-2)* | PPL (C4) | Runtime  |  |
| OHS     | 32       | 34.29       | 32.17    | 17.0 s  | 6.45        | 8.72     | 17.9 min |  |
|         | 128      | 34.26       | 32.14    | 22.6 s  | 6.42        | 8.67     | 20.6 min |  |
| OHS+LLR | 32       | 31.15       | 30.44    | 75.1 s  | 6.14        | 8.55     | 1.24 hr  |  |
|         | 128      | 31.17       | 30.38    | 79.9 s  | 6.09        | 8.52     | 1.30 hr  |  |

<sup>\*</sup> WT-2 refers to the WikiText-2 dataset.


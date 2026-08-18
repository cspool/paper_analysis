# *B. Model Performance*

Perplexity Results. Table [IV](#page-8-0) summarizes the model perplexity of different quantization methods. For KV-cache quantization, on average, P<sup>3</sup> -LLM outperforms Oaken and incurs only < 0.1 perplexity loss. Oaken also has a higher effective KV-cache precision of 4.8 bits due to its inefficient offline calibration, which keeps a large ratio (∼ 10%) of outliers in high precision. On the other hand, P<sup>3</sup> -LLM employs perhead INT4-Asym quantization, where every 128 elements (i.e., the head dimension size of all evaluated models) share a 16-bit scaling factor and a 4-bit zero-point, resulting in an effective precision of 4.16 bits. Hence, P<sup>3</sup> -LLM achieves better perplexity even under a lower precision, demonstrating the effectiveness of our dynamic input-aware smoothing for mitigating outliers. For weight-activation quantization, on average, P<sup>3</sup> -LLM outperforms QuaRot and QoQ with 39% and 22% lower perplexity loss, respectively, despite performing more aggressive quantization with 8-bit queries and attentionscores. This highlights the strength of employing our operanddependent, hybrid numerical formats in preserving model quality under aggressive quantization.

Accuracy Results. Table [V](#page-9-0) shows the accuracy of different quantization methods. On average, P<sup>3</sup> -LLM achieves 1.04% higher accuracy than Oaken under 4-bit KV-cache quantization. Regarding weight-activation quantization, P<sup>3</sup> -LLM improves the average accuracy by a large margin of 2.57% and 3.05% compared to QuaRot and QoQ, respectively. Moreover, the accuracy improvement is particularly pronounced on Llama-3.2-3B. Both QuaRot and QoQ use integer formats that do not fit well to LLM tensor distributions, and rely on calibration datasets that lead to severe overfitting, particularly for smaller models with limited knowledge capacity.

<span id="page-9-0"></span>TABLE V. Accuracy (↑) of reasoning tasks under different quantization methods. Oaken and the top P<sup>3</sup> -LLM use KV4 quantization. QuaRot and QoQ use W4A8KV4 quantization, while the bottom P 3 -LLM uses W4A8KV4P8 quantization.

| Model          |       | Llama-3.1-8B     |       | Llama-3.2-3B |                  |       |       |
|----------------|-------|------------------|-------|--------------|------------------|-------|-------|
| Dataset        |       | MMLU ARC-C GSM8K |       |              | MMLU ARC-C GSM8K |       | Avg.  |
| FP16           | 72.30 | 84.38            | 84.50 | 63.26        | 76.71            | 76.64 | 76.30 |
| Oaken          | 67.31 | 83.70            | 81.80 | 59.93        | 76.79            | 75.66 | 74.20 |
| 3<br>P<br>-LLM | 71.04 | 83.87            | 83.02 | 61.57        | 76.02            | 75.89 | 75.24 |
| QuaRot         | 67.95 | 81.65            | 79.98 | 56.11        | 72.35            | 68.69 | 71.12 |
| QoQ            | 67.46 | 81.31            | 79.53 | 54.11        | 72.18            | 69.22 | 70.64 |
| 3<br>P<br>-LLM | 69.22 | 81.74            | 82.03 | 60.05        | 75.01            | 74.09 | 73.69 |

<span id="page-9-1"></span>TABLE VI. Ablation study on the quantization techniques of P<sup>3</sup> -LLM (gray-shaded). The evaluation metric is Wikitext-2 perplexity (↓).

| Quantization Method             | Llama-2-7B       | Llama-3.1-8B     |  |  |
|---------------------------------|------------------|------------------|--|--|
| Baseline FP16                   | 5.47             | 6.24             |  |  |
| + Pre-RoPE INT4 KV-cache quant  | 5.58<br>(↑ 0.11) | 6.52<br>(↑ 0.28) |  |  |
| + Post-RoPE INT4 KV-cache quant | (↑ 0.03)<br>5.61 | (·)<br>6.52      |  |  |
| → Dynamic key-cache smoothing   | 5.51<br>(↓ 0.10) | 6.35<br>(↓ 0.17) |  |  |
| + INT4 weight quant             | (↑ 0.13)<br>5.64 | (↑ 0.44)<br>6.79 |  |  |
| → 4-bit BitMoD weight quant     | 5.63<br>(↓ 0.01) | 6.72<br>(↓ 0.07) |  |  |
| + FP8-E4M3 attn-score quant     | (↑ 0.04)<br>5.67 | (↑ 0.09)<br>6.81 |  |  |
| → FP8-S0E4M4 attn-score quant   | 5.63<br>(↓ 0.04) | 6.73<br>(↓ 0.08) |  |  |
| + INT8 activation quant         | (↑ 0.09)<br>5.72 | (↑ 0.11)<br>6.84 |  |  |
| → FP8-E4M3 activation quant     | (↓ 0.07)<br>5.65 | (↓ 0.09)<br>6.75 |  |  |

Algorithm Ablation Study. We conduct ablation studies on Llama-2-7B and Llama-3.1-8B to evaluate the accuracy gain of different quantization techniques used in P<sup>3</sup> -LLM. As shown in Table [VI,](#page-9-1) we start with pre-RoPE and post-RoPE KV-cache quantization, and observe noticeable perplexity degradation. This is because both cases do not address the outlier issue of key cache. The proposed dynamic key-cache smoothing significantly improves the perplexity of Llama-2-7B and Llama-3.1-8B by 0.10 and 0.17, respectively. Then, adopting BitMoD for 4-bit weight quantization recovers some perplexity loss of INT4 weights. Subsequently, quantizing attention-scores to the proposed FP8-S0E4M4 format has negligible impact on perplexity. However, applying per-token activation quantization with INT8 brings another major perplexity degradation, which is finally recovered by the adopted FP8-E4M3 activation quantization. To summarize, P<sup>3</sup> -LLM explores optimal numerical formats that better adapt to different operands, thus minimizing their quantization error. In the presence of keycache outliers, P<sup>3</sup> -LLM employs a novel smoothing technique to mitigate their impact.

Analysis of Key-cache Quantization Error. Fig. [8](#page-9-2) presents the normalized key-cache quantization error of Oaken, QoQ, and P<sup>3</sup> -LLM on the Wikitext-2 and C4 datasets across all layers of Llama-2-7B. For each layer, we normalize the error to the average magnitude of all samples in the dataset. Oaken and QoQ rely on Wikitext-2 and Pile [\[18\]](#page-13-30) as calibration datasets to address the key-cache outliers. Although Oaken has similar quantization error as P<sup>3</sup> -LLM on Wikitext-2, its error becomes more pronounced on C4 due to overfitting. QoQ has the highest error since the calibration statistics obtained from Pile does not generalize well to both Wikitext-2 and C4. In contrast, the dynamic input-aware smoothing of P<sup>3</sup> -LLM yields the lowest quantization error, as it does not overfit to any calibration dataset.

<span id="page-9-2"></span>![](_page_9_Figure_6.jpeg)

Fig. 8: Normalized layer-wise key-cache quantization error of Llama-2-7B on Wikitext-2 and C4 datasets. Oaken and QoQ use Wikitext-2 and Pile as calibration datasets, respectively.


# <span id="page-24-0"></span>D.3 ABLATION ON TEMPERATURE t

We evaluate the effect of the temperature parameter t in Equation [7](#page-5-3) on the fine-tuned accuracy of LLaMA-3.2-3B under 4-bit quantization using the parameter budget P(r = 64). The results show that model accuracy remains stable for temperatures between 0.5 and 1.0, while performance slightly degrades when t is too small or too large. A low temperature distributes the parameter budget nearly uniformly across channels, preventing sufficient allocation to channels with large quantization errors. Conversely, an overly high temperature over-concentrates parameters in these large-magnitude channels, neglecting important coefficients within low-magnitude channels and assigning unnecessary parameters within high-magnitude ones. As a result, both excessively low and excessively high temperatures lead to decreased fine-tuned model accuracy. Within the robust range of t ∈ [0.5, 1.0], we selected t = 1 as the default setting, since it naturally favors allocating more parameters to outlier-included, large-magnitude channels while still maintaining stable empirical performance and methodological simplicity.

Table 13: Effect of temperature t on GSM8k accuracy of LLaMA-3.2-3B under 4-bit quantization.

| Temperature t  | 0.25  | 0.5   | 1.0   | 1.5   | 2.0   |
|----------------|-------|-------|-------|-------|-------|
| GSM8k Acc. (%) | 40.11 | 41.39 | 41.47 | 40.64 | 40.04 |

### <span id="page-24-1"></span>D.4 ABLATION ON QUANTIZATION GROUP SIZE

We conduct an ablation study on the effect of quantization group size in the LLaMA-3.2-3B model using 2-bit quantization, where the impact of group size on model accuracy is most clearly observed, as shown in Table [14.](#page-24-2) Smaller group sizes provide finer granularity, leading to higher model accuracy. However, they also incur greater computational overhead during the quantization and dequantization process due to the increased number of quantization parameters. Considering this trade-off, we adopt a group size of 64 for our experiments, consistent with prior works on quantization-aware PEFT.

<span id="page-24-2"></span>Table 14: GSM8k accuracy (%) of QWHA on LLaMA-3.2-3B with 2-bit quantization using various quantization group sizes.

| Group Size | 32    | 64    | 128   | 256   |
|------------|-------|-------|-------|-------|
| Score      | 29.94 | 29.11 | 24.48 | 22.51 |


# <span id="page-19-1"></span>A6 COMPARISONS WITH CLIPPING-BASED METHODS

In this paper, we proposed a novel method, learnable weight clipping (LWC), designed to adaptively determine the weight clipping threshold. LWC sets the threshold by scaling the original minimum and maximum values to delineate the solution space. We compare LWC against existing clippingbased methods: PACT and LSQ. While PACT directly determines the clipping threshold, LSQ focuses on the direct derivation of the scaling factor and zero-point. Both PACT and LSQ were initially formulated as QAT methods, accounting for both weight and activation clipping. For an equitable comparison, our examination is restricted to weight clipping. We integrated PACT and LSQ into our optimization pipeline in lieu of LWC. Table [A14](#page-19-0) illustrates that while PACT and LSQ enhance the performance of weight-only quantization compared to MinMax quantization, their efficacy diminishes in the weight-activation quantization setting. This decline can be attributed to the proposed LET during activation quantization, which alters the weight distribution in each training iteration, undermining the convergence of both LSQ and PACT. In contrast, LWC defines relative scaling values instead of absolute metrics, making it proficient in handling changes in weight distribution. For example, Figure [A5](#page-19-2) shows that LWC can catch the dramatically changing of weights while PACT and LSQ failed.

|  |  | Table A15: Comparisons with SpQR and SqueezeLLM. |
|--|--|--------------------------------------------------|
|  |  |                                                  |

<span id="page-20-1"></span>

| Size        | Method     | Avg bits | Wiki2 | C4   |
|-------------|------------|----------|-------|------|
|             | –          | 16.00    | 5.68  | 7.08 |
|             | SpQR       | 3.94     | 5.87  | 7.28 |
|             | SqueezeLLM | 4.07     | 5.79  | 7.20 |
|             | SqueezeLLM | 4.27     | 5.77  | 7.18 |
| LLaMa-1-7B  | OmniQuant  | 4.16     | 5.77  | 7.21 |
|             | SqueezeLLM | 3.05     | 6.20  | 7.67 |
|             | SqueezeLLM | 3.24     | 6.13  | 7.56 |
|             | OmniQuant  | 3.15     | 6.15  | 7.75 |
|             | –          | 16.00    | 5.09  | 6.61 |
|             | SpQR       | 3.96     | 5.22  | 6.72 |
|             | SqueezeLLM | 4.07     | 5.17  | 6.69 |
|             | SqueezeLLM | 4.26     | 5.17  | 6.68 |
| LLaMa-1-13B | OmniQuant  | 4.16     | 5.17  | 6.69 |
|             | SqueezeLLM | 3.04     | 5.51  | 7.01 |
|             | SqueezeLLM | 3.24     | 5.45  | 6.92 |
|             | OmniQuant  | 3.15     | 5.44  | 7.05 |
|             | –          | 16.00    | 4.10  | 5.98 |
|             | SpQR       | 3.89     | 4.25  | 6.08 |
|             | SqueezeLLM | 4.06     | 4.20  | 6.05 |
|             | SqueezeLLM | 4.25     | 4.18  | 6.04 |
| LLaMa-1-30B | OmniQuant  | 4.16     | 4.19  | 6.06 |
|             | SqueezeLLM | 3.04     | 4.56  | 6.31 |
|             | SqueezeLLM | 3.24     | 4.44  | 6.23 |
|             | OmniQuant  | 3.15     | 4.56  | 6.37 |


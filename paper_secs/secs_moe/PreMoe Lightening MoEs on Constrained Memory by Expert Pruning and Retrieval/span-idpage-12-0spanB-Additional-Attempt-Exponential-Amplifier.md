# <span id="page-12-0"></span>B Additional Attempt: Exponential Amplifier

Complementing the logit transformation choices described in the main text, we also investigated an exponential amplifier as an additional attempt:

$$f_{\exp}(s) = e^{s}. (12)$$

| Table A.1: Generation configurations for each model. |  |  |
|------------------------------------------------------|--|--|
|                                                      |  |  |

| Model           | Temperature | Top-p | Top-K |
|-----------------|-------------|-------|-------|
| DeepSeek-R1     | 0.6         | 0.95  | –     |
| openPangu-Ultra | 0.7         | 1.0   | –     |
| Qwen3-30B-A3B   | 0.6         | 0.95  | 20    |

The motivation is to aggressively enlarge gaps among activated experts, which can be beneficial when many low-confidence activations introduce noise and raw logits contain both negative and positive values.

**Protocol.** We evaluated *f*exp in two settings: (i) applied directly to the top-*K<sup>a</sup>* activated experts without threshold filtering, and (ii) applied after the high-confidence filtering step of our PEU pipeline (i.e., only to logits that pass the local-softmax threshold). Token-level scores are then aggregated as in the PEU calculation to produce expert-level utilities.

<span id="page-13-0"></span>Table B.1: Extended ablation of logit transformations including *e <sup>s</sup>* on accuracy (%) for the DeepSeek-R1 specialist at 50% sparsity.

| Transformation     | Type                                       | MATH-500 | GPQA  | LCB   |  |  |  |
|--------------------|--------------------------------------------|----------|-------|-------|--|--|--|
|                    | Group 1: No Threshold Filtering            |          |       |       |  |  |  |
| s<br>(Raw Logits)  | Baseline                                   | 85.20    | 57.07 | 45.59 |  |  |  |
| sigmoid(s)         | Normalization                              | 95.20    | 46.97 | 23.90 |  |  |  |
| max(s, sigmoid(s)) | Rectifier                                  | 95.00    | 58.08 | 58.09 |  |  |  |
| s<br>e             | Amplifier                                  | 96.80    | 69.70 | 64.34 |  |  |  |
|                    | Group 2: With Threshold Filtering (PreMoE) |          |       |       |  |  |  |
| s<br>(Raw Logits)  | Baseline                                   | 97.00    | 68.18 | 67.28 |  |  |  |
| sigmoid(s)         | Normalization                              | 97.60    | 73.23 | 68.38 |  |  |  |
| max(s, sigmoid(s)) | Rectifier                                  | 97.60    | 72.22 | 66.36 |  |  |  |
| s<br>e             | Amplifier                                  | 97.20    | 73.23 | 66.91 |  |  |  |

**Observations.** Empirically, without threshold filtering, *e <sup>s</sup>* yields the strongest performance by amplifying useful signals in a noisy regime (Table [B.1,](#page-13-0) Group 1). When high-confidence filtering is reinstated, the performance of *e s* converges within a small margin to the top performers (rectifier and/or sigmoid(*s*)), indicating that extreme amplification is unnecessary once noise is suppressed. For robustness and stability across domains and layers, we therefore adopt max(*s*, sigmoid(*s*)) as the default transformation, and include *e <sup>s</sup>* as an additional attempt for completeness.


# Results for expert specialization across datasets.

Figure [5p](#page-12-0)rovides a comprehensive analysis of expert specialization in the Gemma model across all layers and 8 tasks. The results reveal clear specialization trends, with certain experts consistently receiving higher assignment weights for specific tasks. Notably, the middle layers exhibit stronger specialization compared to the front and back layers, where a single expert often dominates task allocation. This indicates a clear task-expert correspondence in these layers.

Backbone dependency across layers and models. Figure [6](#page-13-1) illustrates the weight assigned to the backbone across different layers, models, and datasets, shedding light on the backbone's role in expert specialization. A consistent trend emerges: the backbone LLMs are assigned higher weights in the middle layers, indicating that the backbone primarily handles general knowledge processing in these layers. This aligns with the observation that experts in the middle layers show a strong preference for certain tasks.

This analysis underscores the dynamic interaction between the backbone and experts: *while the backbone leads in processing general knowledge, the surrounding experts diversify their responsibilities, adapting to handle task-specific nuances across datasets*. This balance highlights the flexible architecture of the model, where the backbone ensures stability, and the experts provide specialized capabilities.

Experts diversify over topics. Instance-level routing provides valuable insights into how experts specialize in handling different topics. As illustrated in Figure [7,](#page-13-4) each expert demonstrates distinct preferences for specific topics, reflecting effective specialization. The varying intensity of the routing weights across topics indicates that certain experts

<span id="page-11-0"></span>Table 5: Performance change (%) when randomly masking experts within within specific layers of a fine-tuned Mixture of LoRA at different masking ratios. A 100% masking ratio corresponds to using the backbone only. We report mean and variance reported across 8 commonsense datasets.

| Masked   | Masking Ratio        |                      |                      |                      |                      |  |  |  |  |
|----------|----------------------|----------------------|----------------------|----------------------|----------------------|--|--|--|--|
| Layer(s) | 20%                  | 40%                  | 60%                  | 80%                  | 100%                 |  |  |  |  |
| 1        | $0.00\%_{\pm0.13}$   | $0.03\%_{\pm0.11}$   | $0.00\%_{\pm 0.07}$  | $0.01\%_{\pm 0.09}$  | $-0.14\%_{\pm 0.40}$ |  |  |  |  |
| 2        | $-0.04\%_{\pm 0.09}$ | $0.02\%_{\pm0.11}$   | $-0.03\%_{\pm0.09}$  | $0.00\%_{\pm 0.05}$  | $-0.03\%_{\pm0.08}$  |  |  |  |  |
| 3        | $0.00\%_{\pm 0.05}$  | $-0.02\%_{\pm0.13}$  | $-0.01\%_{\pm0.10}$  | $-0.03\%_{\pm0.12}$  | $0.00\%_{\pm 0.06}$  |  |  |  |  |
| 4        | $0.00\%_{\pm 0.07}$  | $-0.01\%_{\pm0.08}$  | $0.04\%_{\pm 0.08}$  | $0.05\%_{\pm0.08}$   | $0.07\%_{\pm0.10}$   |  |  |  |  |
| 5        | $-0.03\%_{\pm 0.06}$ | $-0.02\%_{\pm0.11}$  | $0.01\%_{\pm 0.08}$  | $-0.01\%_{\pm 0.05}$ | $0.01\%_{\pm 0.04}$  |  |  |  |  |
| 6        | $0.02\%_{\pm 0.03}$  | $-0.05\%_{\pm 0.06}$ | $-0.05\%_{\pm 0.09}$ | $0.02\%_{\pm 0.07}$  | $-0.04\%_{\pm 0.09}$ |  |  |  |  |
| 7        | $-0.02\%_{\pm 0.08}$ | $-0.03\%_{\pm0.09}$  | $0.00\%_{\pm0.04}$   | $-0.02\%_{\pm 0.06}$ | $0.02\%_{\pm 0.05}$  |  |  |  |  |
| 8        | $0.00\%_{\pm0.11}$   | $0.02\%_{\pm 0.08}$  | $-0.03\%_{\pm 0.09}$ | $-0.01\%_{\pm 0.05}$ | $0.04\%_{\pm 0.04}$  |  |  |  |  |
| 9        | $-0.04\%_{\pm 0.08}$ | $-0.01\%_{\pm0.10}$  | $0.00\%_{\pm0.08}$   | $-0.01\%_{\pm 0.07}$ | $0.01\%_{\pm 0.06}$  |  |  |  |  |
| 10       | $-0.01\%_{\pm 0.11}$ | $0.01\%_{\pm 0.08}$  | $0.00\%_{\pm0.10}$   | $-0.02\%_{\pm0.13}$  | $-0.03\%_{\pm0.09}$  |  |  |  |  |
| 11       | $-0.01\%_{\pm 0.09}$ | $-0.01\%_{\pm 0.02}$ | $0.03\%_{\pm0.09}$   | $0.01\%_{\pm 0.10}$  | $-0.01\%_{\pm 0.09}$ |  |  |  |  |
| 12       | $0.00\%_{\pm 0.05}$  | $0.04\%_{\pm0.10}$   | $0.00\%_{\pm0.03}$   | $0.04\%_{\pm 0.09}$  | $-0.02\%_{\pm 0.04}$ |  |  |  |  |
| 13       | $-0.01\%_{\pm 0.08}$ | $0.01\%_{\pm 0.08}$  | $0.02\%_{\pm 0.05}$  | $0.00\%_{\pm0.10}$   | $0.00\%_{\pm 0.05}$  |  |  |  |  |
| 14       | $0.04\%_{\pm 0.07}$  | $0.02\%_{\pm 0.10}$  | $0.06\%_{\pm0.09}$   | $0.01\%_{\pm 0.08}$  | $0.01\%_{\pm 0.09}$  |  |  |  |  |
| 15       | $0.03\%_{\pm0.14}$   | $0.01\%_{\pm 0.06}$  | $0.01\%_{\pm 0.05}$  | $-0.01\%_{\pm0.11}$  | $-0.03\%_{\pm 0.06}$ |  |  |  |  |
| 16       | $0.01\%_{\pm 0.07}$  | $0.03\%_{\pm0.08}$   | $0.00\%_{\pm0.07}$   | $0.00\%_{\pm0.08}$   | $-0.11\%_{\pm 0.37}$ |  |  |  |  |
| 17       | $0.02\%_{\pm 0.09}$  | $0.03\%_{\pm0.05}$   | $0.00\%_{\pm0.10}$   | $0.03\%_{\pm 0.07}$  | $0.01\%_{\pm 0.08}$  |  |  |  |  |
| 18       | $0.00\%_{\pm0.04}$   | $0.00\%_{\pm0.07}$   | $-0.02\%_{\pm0.10}$  | $-0.01\%_{\pm 0.02}$ | $0.01\%_{\pm 0.03}$  |  |  |  |  |
| 19       | $-0.02\%_{\pm 0.04}$ | $0.02\%_{\pm 0.03}$  | $-0.04\%_{\pm0.11}$  | $0.04\%_{\pm 0.04}$  | $-0.02\%_{\pm 0.03}$ |  |  |  |  |
| 20       | $0.03\%_{\pm 0.07}$  | $-0.02\%_{\pm 0.06}$ | $0.01\%_{\pm 0.06}$  | $-0.01\%_{\pm 0.03}$ | $-0.01\%_{\pm 0.05}$ |  |  |  |  |
| 21       | $0.00\%_{\pm0.09}$   | $0.02\%_{\pm 0.04}$  | $0.01\%_{\pm 0.08}$  | $-0.03\%_{\pm0.06}$  | $-0.03\%_{\pm0.09}$  |  |  |  |  |
| 22       | $0.00\%_{\pm 0.04}$  | $0.02\%_{\pm 0.05}$  | $-0.01\%_{\pm 0.02}$ | $-0.01\%_{\pm 0.04}$ | $0.02\%_{\pm 0.04}$  |  |  |  |  |
| 23       | $-0.01\%_{\pm 0.04}$ | $0.01\%_{\pm0.08}$   | $0.01\%_{\pm 0.05}$  | $0.02\%_{\pm 0.04}$  | $0.04\%_{\pm 0.03}$  |  |  |  |  |
| 24       | $0.00\%_{\pm 0.06}$  | $0.01\%_{\pm 0.04}$  | $-0.02\%_{\pm 0.09}$ | $0.00\%_{\pm 0.05}$  | $-0.01\%_{\pm0.11}$  |  |  |  |  |
| 25       | $0.03\%_{\pm0.08}$   | $0.02\%_{\pm 0.05}$  | $-0.01\%_{\pm0.11}$  | $0.03\%_{\pm0.10}$   | $0.01\%_{\pm 0.09}$  |  |  |  |  |
| 26       | $0.02\%_{\pm 0.03}$  | $0.05\%_{\pm 0.06}$  | $0.04\%_{\pm0.10}$   | $0.00\%_{\pm0.05}$   | $-0.02\%_{\pm 0.03}$ |  |  |  |  |
| 27       | $0.02\%_{\pm 0.08}$  | $-0.01\%_{\pm 0.05}$ | $-0.02\%_{\pm 0.06}$ | $0.00\%_{\pm 0.07}$  | $0.04\%_{\pm 0.07}$  |  |  |  |  |
| 28       | $-0.02\%_{\pm 0.07}$ | $-0.07\%_{\pm 0.07}$ | $-0.05\%_{\pm0.07}$  | $0.02\%_{\pm 0.08}$  | $-0.03\%_{\pm0.05}$  |  |  |  |  |
| 29       | $0.01\%_{\pm 0.03}$  | $0.03\%_{\pm0.07}$   | $0.02\%_{\pm0.03}$   | $0.02\%_{\pm0.08}$   | $0.02\%_{\pm 0.05}$  |  |  |  |  |
| 30       | $0.01\%_{\pm 0.05}$  | $0.03\%_{\pm0.10}$   | $0.00\%_{\pm0.08}$   | $0.00\%_{\pm0.04}$   | $0.00\%_{\pm0.07}$   |  |  |  |  |
| 31       | $0.01\%_{\pm 0.04}$  | $-0.04\%_{\pm0.08}$  | $-0.02\%_{\pm0.08}$  | $0.00\%_{\pm0.04}$   | $0.00\%_{\pm0.05}$   |  |  |  |  |
| 32       | $0.00\%_{\pm0.02}$   | $0.00\%_{\pm0.02}$   | $0.00\%_{\pm0.00}$   | $0.00\%_{\pm0.03}$   | $-0.03\%_{\pm0.19}$  |  |  |  |  |

are more suited to specific content areas, while others are more generalized. This diversification showcases the model's ability to dynamically route instances to the most relevant experts, maximizing the efficiency and relevance of the task processing.


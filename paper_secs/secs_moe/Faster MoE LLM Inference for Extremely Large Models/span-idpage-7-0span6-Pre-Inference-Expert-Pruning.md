# <span id="page-7-0"></span>6 Pre-Inference Expert Pruning

Beyond reducing the number of activated experts during inference, we also explore another possibility introduced by fine-grained MoE models, determining which experts to discard before inference begins. In other words, this corresponds to reducing the total number of experts ( $n_e$ ). This approach has already been partially investigated, primarily in the context of Mixtral-series models, where expert pruning is typically guided by information-based metrics or performance comparisons after expert removal.

However, we focus on two additional key questions: 1. To what extent can these methods accelerate inference? 2. Are these methods still effective for fine-grained MoE models? While reducing the total number of experts does not decrease the computational workload, it can increase computational intensity, which could theoretically lead to a net positive effect on inference speed. Furthermore, fine-grained MoE models introduce a unique challenge for global expert reduction, fully randomized initialization, meaning that experts do not exhibit any similarities, making effective global pruning significantly more complex.

#### 6.1 Efficiency

We conducted experiments on DeepSeek V2 with varying  $n_e$  under a concurrent request setting of 512. The results of our evaluation are presented in Table 5.

We observed that the acceleration ratio was also significant at lower throughput levels, with up to 2.3× speedup. This is because when the number of experts is reduced, the computational intensity per expert increases rapidly, leading to a substantial increase in inference speed even with unchanged FLOPs. However, we noticed that at a concurrency level of 192, the throughput after expert reduction decreased. This may be due to a strategy shift within sglang for optimization, potentially triggering an underlying bug. For details on the sglang version used, please refer to Appendix C.


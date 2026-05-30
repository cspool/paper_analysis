# 8 CONCLUSION

In this paper, we target the expert prediction and prefetching problem in MoE LLMs, especially the limitation of crosslayer prediction on the first few layers. We propose a novel same-layer expert prediction using pre-attention activation tensors to solve this problem. Our key observation is that pre-attention activation tensors contains more recent information than the tensors from the previous layer, and matching the ranking of expert selection scores using simple linear functions is possible. Therefore, we propose two lightweight expert predictors with ranking-aware loss functions, eliminating the architectural complexity and cross-layer communication overhead that limits current approaches.

Our proposed pre-attention expert prediction achieves 93.03% exact-match accuracy on DeepSeek V2 Lite, 94.69% on Qwen3, and 97.62% on Phi-mini, showing substantial 15 − 19% absolute improvements over existing cross-layer prediction method FATE [\(Fang et al.,](#page-10-0) [2025\)](#page-10-0) and 30 − 40% improvement compared other prediction and caching methods [\(Zhang et al.,](#page-11-0) [2025b;](#page-11-0) [Tang et al.,](#page-11-0) [2024\)](#page-11-0).

Future MoE architectures can build upon these findings to incorporate native expert prefetching capabilities, while system designers can leverage our parallel execution strategy to optimize inference pipelines across different hardware configurations. Combining high prediction accuracy and reduced system complexity can create more opportunities for MoE system design and deployment strategies.


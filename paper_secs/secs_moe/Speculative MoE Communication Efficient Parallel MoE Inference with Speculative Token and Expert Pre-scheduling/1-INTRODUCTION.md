# 1 INTRODUCTION

The democratization of large language models (LLMs) has been largely driven by continuous model scaling. Over the past five years, the parameter count of the largest trained LLMs has increased by three orders of magnitude, posing significant challenges to the scalability and economic viability of both training and inference under modern AI hardware constraints.

<sup>∗</sup>Equal contribution

<sup>†</sup>Corresponding author

![](_page_1_Figure_1.jpeg)

<span id="page-1-1"></span>Figure 1: Latency breakdown for DeepSeek-V2-Lite inference over a single MoE layer. Hardware: 8-GPU server with fast inter-GPU network (specialized connection with over 400GB/s bandwidth).

To mitigate these challenges, the Mixture-of-Experts (MoE) architecture Fedus et al. (2022); Artetxe et al. (2022); Jiang et al. (2024) has been introduced. Unlike dense models, MoE models sparsely activate one or more expert sub-networks per input, enabling training of trillion-parameter models without compromising accuracy, while maintaining a sub-linear increase in computational cost. This approach has gained widespread adoption in recent industrial-strength LLMs, including DeepSeek-V3 DeepSeek-AI (2024b)/DeepSeek-R1 DeepSeek-AI (2025), GPT-OSS OpenAI et al. (2025), the Qwen3-Series Yang et al. (2025), and Kimi-K2 Team et al. (2025).

However, at inference time, massive MoE models still require substantial GPU/NPU<sup>1</sup> resources to compute, store, and load both expert and attention parameters. To achieve scalability and meet latency requirements, existing inference frameworks deploy multi-dimensional parallelism strategies that distribute experts and attention blocks across interconnected devices. An efficient parallelization scheme must effectively partition input tokens and model parameters, maximize resource utilization, and minimize communication overhead.

To address the memory demands of large-scale MoE deployment and leverage aggregate memory bandwidth, modern inference engines such as SGLang Zheng et al. (2024) and vLLM Kwon et al. (2023) employ expert parallelism (EP), whereby experts are distributed across devices. Attention layers are typically parallelized via data parallelism (DP) or tensor parallelism (TP). While EP enables parallel computation of experts across GPUs, it introduces significant communication overhead: intermediate activations must be *dispatched* from the gating module on a source GPU to the destination GPUs hosting the routed experts, and later *combined* back after expert computation. These operations often result in cluster-wide any-to-any token shuffling, typically implemented via two all2all collective operations (e.g., NCCL/HCCL's all2all).

Our analysis reveals that the inference performance of MoE models remains severely constrained by these costly all2all operations. For instance, a preliminary experiment running SGLang on the DeepSeek-V2-Lite model with 8 GPUs shows that EP communication accounts for up to 59.2% of the forward-pass latency in the MoE layers, respectively—even on high-speed interconnects (see Figure 1). This bottleneck is further exacerbated on slower interconnects such as PCI-e or Ethernet. Therefore, systematically reducing EP communication has become a critical task for improving the efficiency and scalability of MoE inference.

In this paper, we demonstrate that the communication overhead of EP can be substantially reduced through a novel **semantic-aware model—data collaborative scheduling** approach, namely *Semantic Parallelism*. This method forecasts expert routing paths for both requests and individual tokens, and proactively co-schedules tokens and experts to eliminate redundant communication. We implement the above idea in a system **Sem-MoE**, including two key techniques:

First, Sem-MoE performs *offline model scheduling* to reduce expert dispersion. Experts that are frequently activated together are clustered and placed on the same device or server based on predicted token-expert affinities. This grouping is performed periodically offline to avoid runtime overhead.

Second, Sem-MoE employs *online data scheduling* to align input tokens with their corresponding expert groups. This includes: (1) *Inter-request scheduling* for DP-based attention: dynamically batching requests to maximize expert affinity and minimize cross-device transfers. (2) *Intra-request scheduling* for TP-based attention: proactively shuffling token activations during the TP communication phase. Specifically, Sem-MoE replaces the standard post-attention allreduce with a

<span id="page-1-0"></span><sup>&</sup>lt;sup>1</sup>We use GPU and NPU interchangeably in this paper.

shuffled-reduce-scatter and a deferred shuffled-allgather, effectively merging proactive token routing with necessary data transformation.

Through collaborative model-data scheduling, semantic parallelism significantly reduces communication volume and improves inference throughput, as demonstrated through extensive experiments implemented on top of SGLang.

We list the contributions of this paper as follows.

- 1. We conduct a comprehensive data analysis and reveal a significant *context-independent correlation* between tokens and experts in large-scale MoE models, which provides a foundational insight for optimizing expert placement and token routing.
- 2. We design and implement an efficient *model-data collaborative scheduling algorithm* that leverages the observed token–expert affinity. Our scheduler improves local activation rate by 15.4% compared to baseline methods, substantially reducing unnecessary cross-device communication.
- 3. We implement semantic parallelism in Sem-MoE on top of the state-of-the-art inference engine SGLang and perform extensive evaluations. The results demonstrate that Sem-MoE achieves a throughput improvement of up to 2.78x under specific SLOs in Attention-DP scenarios and up to 24.9% latency reduction under Attention-TP setups, validating the practical effectiveness of our approach.


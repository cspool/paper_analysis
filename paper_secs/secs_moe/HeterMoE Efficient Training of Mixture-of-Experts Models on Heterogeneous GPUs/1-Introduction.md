# 1 Introduction

Over the past few years, large language models (LLMs) have demonstrated impressive capabilities in domains like conversation agents [\[3\]](#page-11-0) and coding assistants [\[34,](#page-12-0) [39\]](#page-13-0). Recently, the sparsely-activated Mixture-of-Experts (MoE) architecture has gained popularity as the preferred way of scaling models to hundreds of billions of parameters [\[21,](#page-12-1) [46\]](#page-13-1). MoE models

are often trained with expert parallelism [\[19\]](#page-12-2) where the experts are distributed across GPUs, while token activations are exchanged between GPUs using all-to-all communication.

Meanwhile, GPU hardware is continuously evolving, with newer GPUs increasing performance but simultaneously cost. The demand for newer GPUs is high, with the latest Grok 3 [\[29\]](#page-12-3) and upcoming Llama 4 [\[43\]](#page-13-2) requiring over 100K H100 GPUs (and costing over 4 billion) [\[35\]](#page-13-3). Moreover, the latest GPUs face significant supply constraints, often with a backlog of several months [\[40,](#page-13-4) [41\]](#page-13-5).

Therefore, we need to answer the following question: How can we effectively train MoE-based LLMs on clusters with multiple generations of GPUs? Due to the high cost and limited supplies of latest GPUs, many organizations often retain nodes with older GPUs (e.g., V100) while adding new nodes with the latest GPUs (e.g., H100) [\[1,](#page-11-1) [14\]](#page-12-4) when upgrading infrastructure. Given the extremely high resource requirement for LLM training, we need to utilize all available GPUs.

Leveraging heterogeneous GPUs is challenging due to their varying hardware properties (e.g., memory size, computation capability). To split data, prior work has looked at scaling the batch size in data parallelism for each GPU [\[17,](#page-12-5) [24\]](#page-12-6). To split the model, prior work has explored unevenly distributing model layers in pipeline parallelism across GPUs [\[15,](#page-12-7) [42,](#page-13-6) [45\]](#page-13-7). However, the fundamental limitation of these solutions is that they are agnostic to the existence of heterogeneity *within* the model architecture itself.

Our insight is that different components of MoE models (i.e., attention and expert) exhibit distinct performance characteristics across GPU generations. Older GPUs remain highly efficient for expert computation. In contrast, attention performs significantly better on newer GPUs due to architecturespecific optimizations. For instance, FlashAttention v2 [\[4\]](#page-11-2) exclusively supports Ampere and newer GPUs [\[9\]](#page-11-3), while FlashAttention v3 [\[36\]](#page-13-8) leverages Hopper-specific features like wgmma instructions and TMA [\[28\]](#page-12-8). As experts are already placed across GPUs in expert parallelism, we are presented with an opportunity to assign each GPU only components it can efficiently compute, without introducing additional com-

<sup>∗</sup>Yongji Wu and Xueshen Liu contributed equally.

munication.

In this paper, we present HeterMoE to efficiently train MoE models with heterogeneous GPUs. HeterMoE disaggregates the attention and expert blocks of a transformer layer, assigning them to two different generations of GPUs (newer and older, respectively). The attention-expert disaggregation not only better harvests the compute power of older GPUs, but also alleviates the memory pressure on newer GPUs due to the dominant expert weights and their limited availability.

Still, there are two key challenges HeterMoE must address. First, how can we overlap the computation of different GPUs to reduce the idle wait time of GPUs? Naïve attention-expert disaggregation leaves GPUs spending most of their time waiting for each other due to data dependency. Second, to maximize the extent of overlapping, how can we balance the computation on each GPU at a fine granularity? Simply tuning the degree of parallelism leads to a narrow optimization space, limited by the number of valid configurations. For instance, in expert parallelism, the number of experts must be divisible by the number of GPUs to distribute them.

To address these challenges, we propose zebra parallelism, which divides an input batch to multiple micro-batches and overlaps the attention computation on the newer GPUs and expert computation on the older GPUs of different microbatches. Zebra parallelism differs from pipeline parallelism, where the model is partitioned into multiple GPUs at the granularity of one or more layers. In pipeline parallelism, each sample is sequentially computed on each GPU with the corresponding layers, from the first GPU to the last one. In contrast, in zebra parallelism, the model is partitioned within a single layer, and each sample is computed in a zigzag pattern, passing back and forth between attention (newer) and expert (older) GPUs. To enable fine-grained load balancing of attention and expert GPUs, we propose an asymmetric expert assignment (Asym-EA) mechanism, where we place a part of the experts back to attention GPUs when expert computation is slow. Asym-EA can be selectively activated for a subset of layers and moves back a different number of experts for different layers. We further develop a "gather and squeeze" strategy for Asym-EA to optimize each layer's assignment and minimize the GPU idle time, i.e., bubbles.

We implemented HeterMoE in PyTorch, evaluated across MoE models of different scales, using both an on-premise testbed and EC2 instances under different heterogeneity settings. Our results show that HeterMoE outperforms existing MoE training systems by up to 2.3x and an optimally balanced heterogeneity-aware solution by 1.4x. In addition, HeterMoE achieves 95% training throughput on average compared to a homogeneous setting where all older GPUs are replaced by the newer ones.

We summarize our contributions as follows:

• We observe that the performance disparity between newer and older generation GPUs differs significantly for attention and expert blocks, motivating our solution to disaggregate

<span id="page-1-0"></span>![](_page_1_Figure_7.jpeg)

Figure 1: MoE architecture and expert parallelism.

the two blocks for training MoE models on heterogeneous clusters, with no extra communication.

- We propose zebra parallelism to overlap not only the computation of attention GPUs with the computation of expert GPUs, but also the computation and all-to-all communication on each GPU.
- We design an asymmetric expert assignment mechanism to selectively move different numbers of experts back to attention GPUs for different layers, enabling fine-grained load balancing for zebra parallelism to minimize bubbles.


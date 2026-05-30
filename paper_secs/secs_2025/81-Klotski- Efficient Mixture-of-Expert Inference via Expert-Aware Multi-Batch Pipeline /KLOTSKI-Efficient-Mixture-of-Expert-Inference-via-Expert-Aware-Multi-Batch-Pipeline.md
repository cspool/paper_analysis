# KLOTSKI: Efficient Mixture-of-Expert Inference via Expert-Aware Multi-Batch Pipeline

Zhiyuan Fang Sun Yat-sen University Zhuhai, China fangzhy27@mail2.sysu.edu.cn

Yufeng Lyu Huawei Technologies Co. Ltd Shenzhen, China Ivyufeng1@huawei.com Yuegui Huang Sun Yat-sen University Guangzhou, China huangyg35@mail3.sysu.edu.cn

Wuhui Chen
Sun Yat-sen University
Zhuhai, China
Peng Cheng Laboratory
Shenzhen, China
chenwuh@mail.sysu.edu.cn

Zicong Hong
Hong Kong University of Science and
Technology
Hong Kong, China
ziconghong@gmail.com

Yue Yu\* Peng Cheng Laboratory Shenzhen, China yuy@pcl.ac.cn

Fan Yu Huawei Technologies Co. Ltd Shenzhen, China fan.yu@huawei.com

### **Abstract**

Mixture of Experts (MoE), with its distinctive sparse structure, enables the scaling of language models up to trillions of parameters without significantly increasing computational costs. However, the substantial parameter size presents a challenge for inference, as the expansion in GPU memory cannot keep pace with the growth in parameters. Although offloading techniques utilise memory from the CPU and disk and parallelise the I/O and computation for efficiency, the computation for each expert in MoE models is often less than the I/O, resulting in numerous bubbles in the pipeline.

Therefore, we propose Klotski, an efficient MoE inference engine that significantly reduces pipeline bubbles through a novel *expert-aware multi-batch pipeline* paradigm. The proposed paradigm uses batch processing to extend the computation time of the current layer to overlap with the loading time of the next layer. Although this idea has been effectively applied to dense models, more batches may activate more experts in the MoE, leading to longer loading times and more bubbles. Thus, unlike traditional approaches, we

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. ASPLOS '25, March 30–April 3, 2025, Rotterdam, Netherlands.

© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.

ACM ISBN 979-8-4007-1079-7/25/03 https://doi.org/10.1145/3676641.3716261 Zibin Zheng Sun Yat-sen University Zhuhai, China zhzibin@mail.sysu.edu.cn

balance computation and I/O time and minimise bubbles by orchestrating their inference orders based on their heterogeneous computation and I/O requirements and activation patterns under different batch numbers. Moreover, to adapt to different hardware environments and models, we design a constraint-sensitive I/O-compute planner and a correlation-aware expert prefetcher for a schedule that minimises pipeline bubbles. Experimental results demonstrate that Klotski achieves a superior throughput-latency tradeoff compared to state-of-the-art techniques, with throughput improvements of up to 85.12×.

*CCS Concepts:* • Computing methodologies  $\rightarrow$  Natural language generation.

**Keywords:** Mixture-of-Experts, Offloading, LLM Inference.

#### **ACM Reference Format:**

Zhiyuan Fang, Yuegui Huang, Zicong Hong, Yufeng Lyu, Wuhui Chen, Yue Yu, Fan Yu, and Zibin Zheng. 2025. Klotski: Efficient Mixture-of-Expert Inference via Expert-Aware Multi-Batch Pipeline. In Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS '25), March 30–April 3, 2025, Rotterdam, Netherlands. ACM, New York, NY, USA, 15 pages. https://doi.org/10.1145/3676641.3716261

### 1 Introduction

Owing to the rapid advancement of deep learning, large language models (LLMs) have demonstrated remarkable efficacy across various domains [4, 7, 46]. To facilitate model scalability without escalating the costs associated with training and inference, recent research has introduced sparsely activated

Mixture-of-Experts (MoE) models [10, 33]. MoE models typically replace the Feed-Forward Network (FFN) layers with MoE layers. For each input, only a subset of the parameters (i.e., experts) are sparsely activated for computation, rather than all parameters, significantly reducing computational cost. Current research has demonstrated the superiority of the MoE architecture through extensive experiments [15, 30].

However, due to the skew between model parameter sizes and advances in hardware, MoE-based models, with their massive parameter counts, face more severe memory bottlenecks during inference than other LLMs. For example, DeepSeek-V2 [6], with 236 billion parameters, requires at least seven state-of-the-art (SOTA) GPUs (H100, with 80GB of memory each) for inference. Furthermore, the high cost of memory often makes it difficult to use such large models in more common environments such as personal computers and small servers, limiting the wider adoption of large models [8, 25, 40]. This raises the question of how to deploy MoE models in resource-constrained environments where there is a significant gap between available GPU memory and model parameter sizes.

Offloading is one of the current mainstream solutions for addressing memory optimization during the inference of LLMs [9, 20, 32, 34]. It significantly reduces GPU memory requirements for LLM inference by offloading tensors not needed for the current computation. Applying offloading to MoE models is effective because the experts are sparsely activated, resulting in more parameters that can be offloaded during inference. Recent efforts [9, 43] have proposed offloading strategies tailored for MoE models. Figure 1(a) illustrates the basic paradigm of these methods: prefetching the next layer while computing the current layer to achieve partial overlap of I/O and computation. However, due to the sparse activation of experts, these methods often rely on the accuracy of expert prefetching. For instance, MoE-Infinity [43] performs activation-aware expert prefetching and caching based on expert activation traces. SiDA [8] trains an offline expert predictor in a data-aware manner, achieving a prefetching accuracy of over 90%.

However, significant *inter-layer* and *intra-layer bubbles* (GPU stalls) degrade performance due to the computation and I/O imbalance. Inter-layer bubbles occur because of the imbalance between the attention and expert layers. The large size of experts prevents the computation of the attention layer from sufficiently overlapping with the I/O of the expert layer. In Mixtral-8×7B [19], using an NVIDIA 3090 to process a batch size of 16, the average attention computation is about 2.6 ms, while the single expert transmission time is about 21 ms. Furthermore, when the number of experts selected by the gate exceeds one (as in Mixtral-8×7B and DeepSeekMoE [5], etc.), the I/O overhead for expert transmission multiplies, causing the GPU to wait more frequently. Intra-layer bubbles, on the other hand, result from an imbalance between computation and I/O within the expert layer. In the inference

<span id="page-1-0"></span>![](_page_1_Figure_6.jpeg)

**Figure 1.** Comparison of three kinds of pipeline. We use multiple computations of the current layer to overlap the I/O of the next layer to reduce inter-layer bubbles and adjust the experts' computation order to reduce intra-layer bubbles.

of dense models, the loaded FFN processes all sequences in the batch. However, in MoE models, each activated expert processes only a portion of the sequences in the batch but consume time to transfer multiple FFNs (each expert is an FFN). For instance, processing a token with a single expert in Mixtral-8×7B takes less than 1 ms, which is much less than the transmission delays. This leads to substantial intra-layer bubbles between the computations of multiple experts.

Inspired by related work on dense models [34, 41], a straight-forward approach is to consider the computations of multiple batches simultaneously. This increases the total computation time, thereby allowing for the overlap of the I/O time for the next layer. Specifically, after loading the weights of a layer, they are shared across multiple batches, allowing consecutive computations within the current layer. This provides sufficient time for loading the weights of the subsequent layer, thereby significantly reducing inter-layer bubbles.

Despite this, considering the computations of multiple batches simultaneously also means increasing the diversity of the inputs to the MoE layer. Given the sensitivity of the gating mechanism to data variability [24, 42], the total number of activated experts may increase. As shown in Figure 1(b), in addition to the experts activated in Figure 1(a), experts 5 and 3 are also activated. Although multiple computations in the attention layer can overlap the I/O of some experts, more experts are activated, resulting in more intra-layer bubbles in the pipeline, due to the long I/O time for these experts.

To tackle this challenge, we propose an *expert-aware multi-batch pipeline* paradigm. Specifically, based on current observations [23, 42], there is a phenomenon in MoE inference where a few experts handle the majority of tokens, referred to as *hot experts*. Correspondingly, other experts are termed

cold experts. Considering a large number of tokens across multiple batches, hot experts exhibit high computational demand and low I/O demand, while cold experts exhibit the opposite. By leveraging this complementary relationship, we can overlap the high I/O demand of cold experts with the high computational demand of hot experts, effectively minimizing intra-layer bubbles between experts. As illustrated in [Figure 1\(](#page-1-0)c), we prefetch only the hot experts 2 and 4 and partition the computations of multiple batches by experts rather than by batches. Furthermore, we adjust the computation order of the experts, prioritizing the substantial computations of hot experts 2 and 4, providing more ample time for the transmission of cold experts 5, 3, and 1. This effectively compresses the intra-layer bubbles.

In this paper, based on the above paradigm, we propose Klotski, an MoE-oriented inference engine that can perform high-throughput inference in resource-constrained environments, achieving inference pipeline with near-zero bubble, as shown in [Figure 1\(](#page-1-0)c). To summarize, we make the following contributions:

- We propose an expert-aware multi-batch pipeline paradigm that leverages the high computational demand and low I/O demand of hot experts to orchestrate multibatch computations, aiming to minimize both interlayer and intra-layer bubbles.
- We design a constraint-sensitive IO-compute planner to formulate execution plans for this paradigm in various environments.
- We propose adaptive tensor placement and a correlationaware expert prefetcher, enabling appropriate offloading and prefetching when dealing with different storage resources and MoE models.
- We implement the above strategies in Klotski, an MoE-oriented inference engine, which enables highthroughput inference of MoE with offloading.
- To evaluate Klotski, we compare it with Accelerate [\[13\]](#page-13-15), Deepspeed-FastGen [\[16\]](#page-13-16), FlexGen [\[34\]](#page-14-4), MoE-Infinity [\[43\]](#page-14-5), and Fiddler [\[20\]](#page-13-9). The experimental results demonstrate that Klotski can make inference of MoE more efficiently, and achieve 85.12×, 15.45×, 2.23×, 19.06×, and 9.53× throughput improvement than that of the three aforementioned works, respectively.


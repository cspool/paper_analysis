# MoEsaic: Shared Mixture of Experts

Umesh Deshpande, Travis Janssen Mudhakar Srivatsa, Swaminathan Sundararaman IBM Research, USA

#### Abstract

Mixture of Expert (MoE) models consist of several experts, each specializing in a specific task. During inference, a subset of the experts is invoked based on their relevance to the request. MoE's modular architecture lets users compose their model from popular off-the-shelf experts. This leads to multiple MoE deployments with identical experts. The duplication of experts across model instances results in excessive GPU memory consumption and increased model serving cost. Moreover, since all experts are not invoked for each request, individual experts rarely receive enough requests to exploit the GPUs' computational capabilities, resulting in low GPU utilization. To address these problems, we propose Shared Mixture of Experts in MoEsaic. MoEsaic automatically identifies and deduplicates identical experts across model instances, thus reducing their memory footprint. Moreover, it batches the requests directed toward the identical experts belonging to different clients, which also improves the processing efficiency. We show that for Mixtral-8x7B model, when compared to deploying dedicated MoE instances, MoEsaic can serve 7X more model instances with little impact on inference performance.

#### CCS Concepts

• Computing methodologies → Planning with abstraction and generalization.

#### Keywords

Mixture of Experts, Model Sharing

#### ACM Reference Format:

Umesh Deshpande, Travis Janssen and Mudhakar Srivatsa, Swaminathan Sundararaman. 2024. MoEsaic: Shared Mixture of Experts. In ACM Symposium on Cloud Computing (SoCC '24), November 20– 22, 2024, Redmond, WA, USA. ACM, New York, NY, USA, [9](#page-8-0) pages. <https://doi.org/10.1145/3698038.3698521>

![](_page_0_Picture_10.jpeg)

[This work is licensed under a Creative Commons Attribution-](https://creativecommons.org/licenses/by-nc-sa/4.0/)NonCommercial-ShareAlike International 4.0 License. SoCC '24, November 20–22, 2024, Redmond, WA, USA © 2024 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-1286-9/24/11 <https://doi.org/10.1145/3698038.3698521>

#### 1 Introduction

The power-law observed in language models suggests that a model's performance improves as the number of its parameters increases [\[16\]](#page-7-0). This has driven the development of Large Language Models (LLMs) that possess hundreds of billions of parameters. However, to manage the computational complexity resulting from such large sizes, routing networks like Sparse Mixture of Experts (MoE) [\[28\]](#page-8-1) are employed. These networks effectively reduce the computational load by selectively interacting with only a subset of the parameters, referred to as experts. In MoE, the invocation of experts is governed by a gating mechanism that invokes the most relevant experts to the request. This architecture is particularly useful for handling diverse datasets, where a single model may struggle to capture all underlying patterns. MoE's superior generative capability and performance have made it the preferred architecture for LLMs [\[7,](#page-7-1) [15,](#page-7-2) [18\]](#page-7-3). As a result, MoE has been adopted by well-known Large Language Models (LLMs), such as Grok [\[4\]](#page-7-4) and Gemini [\[31\]](#page-8-2).

MoEs are inherently modular, often consisting of a combination of purpose-built experts. For instance, multi-modal inference benefits from a combination of modality-specific experts [\[8,](#page-7-5) [23,](#page-8-3) [26,](#page-8-4) [27,](#page-8-5) [34\]](#page-8-6), e.g., a text expert vs. an image expert. Experts can be added or removed from an MoE model to only serve the desired tasks [\[11\]](#page-7-6). Several emerging tools have allowed users to compose a new MoE by combining the off-the-shelf models [\[1,](#page-7-7) [3,](#page-7-8) [19,](#page-8-7) [30\]](#page-8-8). Moreover, users can fine-tune only a subset of task-specific experts [\[32\]](#page-8-9) in tens of minutes. These approaches eliminate the necessity of retraining the entire model from scratch. However, these advantages of MoE have also created the following challenges in a multi-client environment.

First, due to the reuse of the off-the-shelf experts or finetuning of selective experts, identical experts can be found across different model instances. Despite the identical experts, different clients running variants of the same MoE model need to run dedicated instances of their model. As a result, identical experts from different variants occupy distinct GPU memory space. For instance, even in common MoE models, e.g., Mixtral 8x7B, each expert consumes 14GB of GPU memory. Figure [1](#page-2-0) (a) demonstrates the problem with 2 variants (e.g., belonging to 2 clients). The Second challenge with MoE is that it exhibits sparse execution patterns compared to regular neural network models. Specifically, because

MoE selects only a subset of experts for each request, only a fraction of the parameters of the model are involved in serving a request. Because of the higher memory-to-compute ratio, MoE exhausts GPU memory before fully utilizing its computational capacity, which leads to GPU underutilization [\[21\]](#page-8-10). This problem is exacerbated in a multi-client environment, where individual model instances do not receive enough requests to form large batches [\[29\]](#page-8-11) and benefit from GPUs' capability to efficiently process multiple requests in parallel.

The existing techniques that optimize MoEs focus on either reducing the memory footprint of MoEs through parameter pruning, quantization, removing experts [\[14,](#page-7-9) [20\]](#page-8-12) or specifically designing MoEs with fewer parameters [\[12\]](#page-7-10). Our approach is orthogonal to such optimizations and through deduplication it can achieve an order of magnitude saving without altering the accuracy or behavior of the constituent MoEs.

In this paper, we propose shared Mixture of Experts (MoEsaic) to address the above problems. With MoEsaic, a single inference service can host several MoE model instances. MoEsaic identifies and deduplicates identical experts across MoE instances. Figure [1](#page-2-0) (b) shows an instance of MoEsaic deduplicated experts across two clients. The sharing of experts is beneficial in the following ways. First, the sharing reduces the GPU memory requirement for running multiple model instances. Second, the requests for the deduplicated experts are batched, even if they belong to different clients. For instance, in the Figure [1](#page-2-0) (b) requests R1 and R2 (headed towards expert A), belong to two different clients, however batching their execution with expert A improves the overall computational efficiency.

The contributions of MoEsaic are as follows.

- In MoEsaic, we present an approach to transparently detect identical experts from across multiple clients and share them during inference. The client need not provide additional hints to enable sharing. Moreover, to support the large experts that cannot fit in the memory of a single GPU, MoEsaic supports tensor-parallel execution of experts. In a tensor-parallel mode, MoEsaic deduplicates shards of experts to enable the sharing of experts across multiple GPUs.
- We present techniques for improving the processing efficiency while leveraging model sharing in MoE. Specifically, MoEsaic batches the execution of requests from multiple clients when they are to be executed by the deduplicated experts. We demonstrate that such batching improves computational efficiency. Next, we fuse the gates of multiple MoE model instances to avoid sequential invocation of gates during inference at each

- layer. The fused routing of requests further improves the scalability of MoEsaic.
- We have implemented MoEsaic in vLLM. Additional experts can be added to an existing model using a familiar interface that mimics the addition of LoRA adapters. Moreover, MoEsaic allows service providers to dynamically integrate new clients, namely their experts and gates, to an already deployed MoE instance(s).

#### 2 Related Work


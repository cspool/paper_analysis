# Abstract

Mixture-of-Experts (MoE) based large language models (LLMs) have demonstrated exceptional performance across a wide range of downstream tasks and application scenarios. Recent advancements in MoE-based LLMs, such as Deepseek MoE, incorporate fine-grained expert segmentation and shared expert isolation to unlock greater potential for expert specialization. While this technique significantly enhances model capability and reduces training costs, it introduces challenges related to increased inference latency and reduced throughput.

To address these challenges, we propose IFMoE (Inference Framework for Finegrained MoE), a system specifically designed to enhance the inference performance of fine-grained MoE models. IFMoE introduces a redesigned parallelism mechanism tailored for MoE inference and incorporates the concept of Speculative Decoding to alleviate the high latency introduced by expert fusion kernel calculation. Although it is not an entirely lossless method, experiments demonstrate that IFMoE maintains downstream performance while achieving a 30% improvement in both inference latency and throughput.


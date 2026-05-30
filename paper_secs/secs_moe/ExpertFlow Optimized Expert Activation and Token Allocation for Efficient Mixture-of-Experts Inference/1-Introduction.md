# 1 Introduction

Sparse MoE models [5, 8, 14, 25] scale parameter size efficiently by activating only a small subset of experts per input, reducing pertoken computation while keeping accuracy comparable to dense LLMs [10, 21, 22]. This efficiency, however, increases memory usage. For instance, Mixtral-8×7B [14] requires more than 96 GB of GPU memory, exceeding the 80 GB capacity of an NVIDIA A100 GPU. Although 45.1B of its 46.7B parameters belong to expert modules, only a small fraction is used per input, leading to substantial memory redundancy. This sparsity suggests that offloading inactive experts to CPU and loading only the needed ones can reduce GPU memory demand. Existing studies have investigated such offloading methods [6, 7, 12, 15, 27, 35], but remain limited by three challenges.

Inefficient expert prediction. Early and accurate expert activation prediction underpins effective offloading, as it enables scheduling and prefetching before experts are required. Prior work takes two routes. Regression-based methods [6, 12] approximate router scores, but even small score errors can affect output quality, necessitating extensive fine-tuning to recover the original routing. Classification-based methods predict selected experts directly. Heuristic variants based on token-expert statistics [14, 17, 34] are lightweight but fail to capture input-dependent routing behavior. Learning-based predictors (e.g., ProMoE [27]) improve accuracy, yet their layer-by-layer sequential design reveals expert usage only after the previous layer executes, restricting scheduling flexibility.

Low expert utilization. In the decoding phase, the token distribution across experts can be highly imbalanced, and some experts may receive only a single token. Since expert kernels have near-constant cost when handling a small number of tokens [32], such sparse assignments lead to low compute efficiency.

Ineffective expert caching. Expert caching is central to controlling GPU memory usage. The commonly used LRU policy [7] evicts experts purely by recency and overlooks routing patterns, leading to unstable cache hit rates under MoE's dynamic activations. SE-MoE [35] improves locality by caching all experts from two consecutive layers through a ring-buffer design, but this creates large memory overhead for models with many experts (e.g., Switch-128) and repeatedly loads inactive experts, resulting in unnecessary CPU–GPU transfers.

To address these challenges under resource-constrained settings such as single-GPU inference, we propose *ExpertFlow*, a unified system for memory-efficient MoE execution. Fig. 1 shows an example where tokens from two batches activate different experts across layers, leading to fragmented execution and high memory usage when processed directly. *ExpertFlow* recasts this process as a predictive and coordinated pipeline through three components: ① the **Routing Path Predictor (RPP)** predicts expert activations

<span id="page-1-0"></span>![](_page_1_Figure_2.jpeg)

Figure 1: Overview of ExpertFlow. Given two input batches, ① Routing Path Predictor predicts expert activation across all layers for all tokens, ② Token Scheduler uses the prediction to reorder tokens across batches to consolidate expert usage, ③ Expert Cache Engine preloads only required experts into GPU from CPU, and ④ the MoE model executes with optimized token flow and heterogeneous expert placement.

for all tokens and all layers in one forward pass, providing early global routing signals; ② the **Token Scheduler (TS)** reorganizes tokens based on predicted paths to consolidate expert usage and increase compute efficiency; ③ the **Expert Cache Engine (ECE)** loads only the needed experts into GPU memory and reuses them across steps, with lightweight correction for mispredictions. Our contributions are summarized as follows:

- We identify three core bottlenecks in MoE offloading: inefficient expert prediction, low expert utilization, and ineffective caching under dynamic routing.
- We introduce ExpertFlow, a unified system that integrates predictive scheduling, routing-aware token rebatching, and adaptive caching with lightweight correction. ExpertFlow reduces GPU memory usage by up to 93.72% and improves throughput by up to 10× over strong offloading methods, enabling efficient MoE inference on constrained single-GPU settings.
- Our *RPP* achieves up to 95% expert prediction accuracy with strong cross-domain generalization. The *TS* improves throughput by up to 16.19% via enhanced expert reuse. The *ECE* attains a cache hit ratio of 91.96%, outperforming LRU by up to 61.15%.

## 2 Related Work

## 2.1 Mixture-of-Experts (MoE)

MoE models [13] improve scalability by activating only a subset of experts for each token through a softmax-based gating mechanism, where the gating score for experts is  $G(x) = \operatorname{softmax}(xW_g)$  and the model selects the top-k experts with the highest scores. The MoE layer output is then computed as a weighted sum of the selected experts,  $y = \sum_{i \in \operatorname{TopK}(G(x))} G_i(x) E_i(x)$ . With advances in hardware

and training methods, transformer-based MoE architectures have become widely used and show strong performance across many tasks [5, 8, 25, 29], where the gating function determines which experts each token activates and shapes the routing pattern that drives system efficiency.

## 2.2 Model Compression

LLM inference faces substantial GPU memory constraints, prompting prior research to explore a range of solutions. Distillation techniques [5, 35] reduce the number of experts by compressing the teacher network into a smaller student network. Model pruning methods have also been explored, such as pruning non-essential experts during fine-tuning based on usage frequency [4] and merging similar experts followed by low-rank decomposition [16]. Post-training quantization [7, 9, 18, 19] further reduces memory consumption by converting pre-trained models to lower-precision ones (e.g., Int4), without requiring extensive retraining. The contribution of our proposed *ExpertFlow* is orthogonal to this direction, and *ExpertFlow* can be seamlessly integrated with these techniques to further reduce GPU memory cost during MoE inference.

## 2.3 Model Offloading

Model offloading reduces GPU memory usage by moving model states or computations to cheaper storage or processing units. Early work such as ZeRO [23, 24] offloaded optimizer states, gradients, and weights to CPUs or SSDs during training, and later extensions applied similar ideas to inference [3, 26, 28]. FlexGen [26] uses a zigzag block schedule to offload activations and KV caches, allowing large models like OPT-175B [37] to run on a single 16GB GPU, while Lamina [3] improves efficiency by shifting attention computation to CPUs. However, these methods are designed for dense LLMs and do not handle the dynamic, input-dependent routing of MoE models. Existing MoE offloading approaches either rely on low-accuracy heuristics [7, 17] or require costly predictor training [6, 12], limiting practical adoption. In contrast, we develop a unified system that provides accurate and low-cost expert routing prediction, enabling more efficient and flexible MoE inference.

## 3 Method


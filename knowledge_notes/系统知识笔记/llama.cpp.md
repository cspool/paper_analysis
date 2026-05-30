## llama.cpp

术语解释
llama.cpp 是一个开源的高性能 LLM 推理框架，使用 C/C++ 编写，支持 CPU 和 GPU（CUDA/Metal/Vulkan）后端，以 GGUF 量化格式高效部署 LLM 和 MoE 模型。BuddyMoE 以及多个 MoE offloading 系统（MoE-APEX、Mixtral-Offloading）均基于 llama.cpp 构建。

术语是什么？
llama.cpp (https://github.com/ggerganov/llama.cpp) 由 Georgi Gerganov 创建，核心特性包括：(1) 纯 C/C++ 实现，无需 Python 依赖；(2) 支持多种量化格式（GGUF: Q4_0, Q4_K_M, Q8_0 等）；(3) CUDA/Metal/Vulkan GPU 后端加速；(4) 内置 KV cache 管理和 continuous batching；(5) 对 MoE 模型提供基础支持（含 expert offloading 和 server 模式）。在 MoE 推理中，llama.cpp 提供原生 CPU offloading 机制允许 large expert 参数从 GPU 卸载到 CPU memory，以及 CUDA backend 执行 GPU-resident expert 的 FFN GEMM kernel。

从系统架构角度拆解术语：
在 BuddyMoE 中，llama.cpp 作为底层推理引擎，其 MoE pipeline 为：token → Attention (CUDA) → Router (CUDA) → Expert residence check → if GPU: FFN GEMM (CUDA) else if BuddyMoE: Buddy substitution (CUDA kernel) → Weighted combine。BuddyMoE 在 llama.cpp 的 router 和 expert execution 之间插入 buddy replacement runtime 层，不修改原始 router 和 expert 权重。

术语一般如何实现？如何使用？
- 代码：C/C++ 核心 + CUDA kernels + Python bindings (llama-cpp-python)
- MoE 使用场景：使用 `--num-experts` 和 `--expert-offloading` 标志控制，GGUF 格式存储 MoE 模型权重
- BuddyMoE 修改：在 llama.cpp 推理 pipeline 中插入 buddy replacement CUDA kernel，利用 atomic CAS 实现并行替换
- 相关 MoE 系统：MoE-APEX（基于 llama.cpp + adaptive precision + expert prefetching）、Mixtral-Offloading（基于 llama.cpp + LRU expert caching）

涉及论文标题：
- BuddyMoE Exploiting Expert Redundancy to Accelerate Memory-Constrained Mixture-of-Experts Inference

---

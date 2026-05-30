## Reactive Caching for MoE（MoE的被动式缓存）

术语是什么？
Reactive Caching 是现有 MoE expert offloading 系统中的传统缓存策略：expert 参数大部分存储在 CPU memory，GPU memory 中维护有限大小的 expert cache（LRU 或 static）。当推理中 gate function 选择了一个不在 GPU cache 中的 expert 时，触发 cache miss——系统被动地从 CPU memory 加载该 expert（cudaMemcpy），推理计算被阻塞直到传输完成。ProMoE 测量表明 DS-1 50% cache rate 下 decode 阶段 60.4% 时间、prefill 阶段 82.7% 时间用于等待 expert 加载；llama.cpp 更严重（prefill 94.2%, decode 79.0%）。Modern decoder-only MoE 的 uniform expert access pattern 进一步限制了 LRU cache hit rate。

从系统架构角度拆解术语：
Reactive Caching 执行流程：gate function 选择 experts → 遍历每个 expert 检查是否在 GPU cache 中 → 命中则直接计算，未命中则 cudaMemcpy(CPU→GPU) 阻塞等待 → expert FFN 计算 → LRU update。核心问题：cache miss 时的 cudaMemcpy 完全暴露在推理关键路径上，GPU 在等待 PCIe 传输期间闲置。

术语一般如何实现？如何使用：
典型实现包括 LRU cache（Mixtral-offloading）、static cache（固定 Top-N experts + 缓冲 buffer）、CUDA Unified Memory（页级自动迁移但有传输放大）。优势是实现简单，劣势是 cache miss 延迟完全暴露在关键路径。ProMoE 的 proactive caching 是对此的改进——通过预测将数据传输移出关键路径。

涉及论文标题：
- ProMoE: Fast MoE-based LLM Serving using Proactive Caching

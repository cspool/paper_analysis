## MoE Offloading for Single-GPU Inference（单GPU MoE模型卸载推理）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MoE Offloading 是一种在单 GPU 上运行超大 MoE 模型（远超 GPU 显存容量）的内存管理技术。核心设计是将完整的模型参数和 KV-cache 存储在容量更大、成本更低的 host memory（CPU DRAM）中，仅当 GPU 需要计算时，才将所需的参数和 KV-cache 传输到 GPU 显存。MoE 模型的天然稀疏性（每个 token 仅激活 k 个 expert）使 offloading 特别适合：无需将所有 expert 参数同时驻留在 GPU 中，只需按需加载被激活的 expert。MoE offloading 系统通常管理两级内存：GPU memory（用于计算和快速数据访问，包含 resident store 和 staging buffer）和 CPU memory（存储完整模型权重和 KV-cache）。当 GPU 计算需要尚未在 resident store 中的 weight/KV-cache 时，要么提前预取（overlap with computation），要么 on-demand fetch（GPU stall 等待传输）。在 offloading 场景下，PCIe HtoD 带宽（如 PCIe 4.0 ×16 约 32 GB/s）成为关键瓶颈，因此系统设计需最大化 computation-memory copy overlap。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
MoE offloading 系统的典型执行流程（以 MoE-GEN 为例）：

```
单层 MoE offloading 推理（解码阶段）:
┌─────────────────────────────────────────────────────────┐
│ Host Memory (CPU DRAM)                                   │
│  - 完整 model weights (all experts, attention)           │
│  - 完整 KV-cache (所有 layers, 所有 tokens)              │
│  - 中间 activation states                                │
└──────────────┬──────────────────────────────────────────┘
               │ PCIe 4.0 (32 GB/s HtoD, ~32 GB/s DtoH)
┌──────────────▼──────────────────────────────────────────┐
│ GPU Memory (24-48GB)                                     │
│  - S_Dense buffer: 预取当前 dense module weights         │
│  - S_Expert buffer: 预取下一个 expert weights            │
│  - S_Params: 缓存的常驻模型参数（可选）                  │
│  - KV-cache buffer: 当前 attention batch 的 KV-cache     │
│  - S_IS: intermediate states (QKV projections 等)        │
│  GPU 计算: attention micro-batches → expert 大 batch     │
└─────────────────────────────────────────────────────────┘
```

关键设计权衡：
- **Partial vs Full KV-cache offloading**：部分卸载（保留部分 KV-cache 在 GPU）可减少 KV-cache HtoD copy，但挤压 expert batch size。MoE-GEN 选择 full offloading，在大型 dataset 上节省最高 20× 的 expert weight fetching 流量。
- **CPU computation offloading**：将 attention 计算分流到 CPU，不仅利用 CPU 算力，更重要的是减少 KV-cache HtoD copy 对 PCIe 带宽的竞争，将带宽让给 expert weight 预取。
- **Overlap 策略**：HtoD engine 在 GPU 计算当前 module 时异步预取下一个 module 的 weights；DtoH engine 异步将新生成的 KV-cache 写回 host memory。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现 MoE offloading 的主要系统和方案：
1. **FlexGen**：以 GPU memory 为约束，通过线性规划搜索最优的 offloading 和 recomputation 策略，按轮次重用已加载的权重进行多次 forward pass。不支持 MoE 特殊优化。
2. **DeepSpeed-Inference**：将 MoE layer 视为 dense MLP 处理，支持 ZeRO-Inference 的 layer-wise weight offloading。
3. **MoE-Lightning**：在 FlexGen 基础上优化 GPU-CPU-I/O overlap，使用 profiling 预先规划 memory movement schedule，但保留 model-based batching。
4. **llama.cpp (Ollama)**：支持 `--n-gpu-layers` 和 `--n-cpu-moe` 参数，将 expert tensors 按 pattern（如 `.ffn_.*_exps.`）识别并 offload 到 CPU，attention layers 保留在 GPU。
5. **KTransformers**：AMX-optimized CPU kernels（21.3 TFLOPS/socket），专家延迟执行（Expert Deferral）使 CPU expert compute 与 GPU attention 重叠，集成到 SGLang。
6. **MoE-GEN**：module-based batching + full KV-cache offloading + CPU attention + DAG-based batch size search，针对 offline high-throughput 场景。

涉及论文标题：
- MoE-Gen: High-Throughput MoE Inference on a Single GPU with Module-Based Batching

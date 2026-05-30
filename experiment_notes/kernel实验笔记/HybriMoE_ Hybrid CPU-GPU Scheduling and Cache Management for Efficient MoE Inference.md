## HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management for Efficient MoE Inference

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - HybriMoE 在 kernel 调度/运行时计算层面的核心实现：
    1. **Modified C++ Expert Computation Kernels**：修改 llama.cpp 的 C++ kernels，使其直接处理 expert 计算任务分配（从 HybriMoE scheduler 接收 CPU/GPU 设备分配决策），消除冗余的 Python 调用开销。CPU 端 expert FFN 计算使用 llama.cpp 优化的 C++ GEMM kernel（利用 CPU 大 cache 实现内存访问与计算重叠）。
    2. **Multi-Stream CUDA Parallel Execution Engine**：利用 fine-grained CUDA stream 调度实现三路并行——Stream 0 (GPU compute: expert FFN GEMM via Marlin 4-bit quantization kernel)、Stream 1 (PCIe transfer: cudaMemcpyAsync CPU↔GPU expert weight copy)、CPU Thread Pool (CPU compute: llama.cpp C++ expert FFN kernel)。通过 CUDA event 同步不同 stream 间的数据依赖。
    3. **Marlin Quantization Kernel Integration**：集成 llama.cpp 的 Marlin 4-bit 量化 kernel（SOTA 4-bit GPU GEMM kernel），显著提升 GPU 端 expert FFN 计算效率和降低 GPU memory footprint。
    4. **Simulation-Based Scheduling Kernel**：在 warmup 阶段 profiling CPU/GPU computation latency 和 PCIe transfer latency 之后，每次 MoE layer 执行前运行轻量级仿真（贪心 fill timelines），输出 expert-to-device 分配计划交给运行时 kernel 调度器执行。
  - 实验比较：完整系统 vs llama.cpp（静态 CPU/GPU 映射）、AdapMoE（GPU-centric 调度）、kTransformers（CPU-GPU hybrid 调度）。消融对比 Scheduling/Prefetching/Caching 各组件的 kernel 级时间贡献。

- 后端平台是什么，配置是什么。
  - GPU 后端：NVIDIA RTX A6000（Ampere，支持 CUDA stream 并行和 Marlin 4-bit kernel）
  - CPU 后端：Intel Xeon Gold 5220R（10 cores，用于模拟边缘部署场景）
  - CPU-GPU 传输：PCIe（用于 expert weight/activation 的 CPU↔GPU 数据传输）
  - Expert 量化：Marlin 4-bit quantization（GPU GEMM）+ llama.cpp CPU GEMM

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 **kTransformers** (https://github.com/kvcache-ai/ktransformers) + **llama.cpp** (https://github.com/ggerganov/llama.cpp) kernels。
  - 修改内容：
    - **C++ Expert Kernels**：修改 llama.cpp 的 expert layer C++ kernel，接收外部传入的 device assignment（CPU or GPU），消除 Python dispatch 开销。CPU kernel 利用 CPU 大 cache 实现首 expert 后的内存访问-计算重叠。
    - **Marlin 4-bit Quantization Kernel**：在 GPU 端 expert FFN 中使用 Marlin kernel 替代标准 FP16 cuBLAS GEMM，减少内存占用和提升吞吐。
    - **Multi-Stream Execution**：初始化 2 个 CUDA stream（GPU compute + PCIe transfer）+ CPU thread pool，通过 CUDA event 同步 data dependency。
    - **Simulation Scheduler**：warmup 阶段 profiling 各操作延迟 → 运行时每层执行前运行贪心仿真（<μs 级开销）→ 输出 task-to-device schedule。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - **开源情况**：代码开源在 https://github.com/PKU-SEC-Lab/HybriMoE
  - **Multi-Stream Kernel 执行原理全过程**：

```
┌── Kernel Input ──────────────────────────────────────────────┐
│ Per MoE layer, per iteration:                                 │
│   activated_experts = [(expert_id, num_tokens, in_cache)]     │
│   Example: [(E₁, 128, True), (E₂, 32, False)]               │
│   warmup_profiled:                                            │
│     cpu_latency_per_token = α (e.g. 0.05ms/token)            │
│     gpu_latency = β (nearly constant, e.g. 3ms)              │
│     pcie_transfer_latency_per_expert = γ (e.g. 5ms)          │
└──────────────────────────────────────────────────────────────┘

┌── Simulation Scheduler (CPU, per-layer, <μs latency) ────────┐
│ // 构建优先级队列                                               │
│ GPU_Queue = sort(cached_experts, by num_tokens desc)          │
│ CPU_Queue = sort(uncached_experts, by num_tokens asc)         │
│                                                                │
│ // 初始化三条 timeline                                          │
│ timeline_GPU = 0, timeline_CPU = 0, timeline_PCIe = 0         │
│                                                                │
│ while GPU_Queue or CPU_Queue:                                  │
│     t_next = min(timeline_GPU, timeline_CPU, timeline_PCIe)   │
│     if t_next == timeline_GPU and GPU_Queue:                   │
│         e = GPU_Queue.pop()  // 最高负载                        │
│         timeline_GPU += gpu_latency  // ≈ constant             │
│         // 同时检查 CPU Queue: 若 CPU idle 且有 cached expert   │
│         // 可从 GPU Queue 窃取 (低负载优先)                      │
│     elif t_next == timeline_CPU and CPU_Queue:                 │
│         e = CPU_Queue.pop()  // 最低负载                        │
│         timeline_CPU += e.num_tokens × cpu_latency_per_token   │
│         // 首 expert 在 CPU 上较慢 (cold cache),               │
│         // 后续 expert 因 CPU cache 命中而加速                  │
│     elif t_next == timeline_PCIe and CPU_Queue:                │
│         e = CPU_Queue.pop_highest_load()  // 最高负载           │
│         timeline_PCIe += pcie_transfer_latency                 │
│         GPU_Queue.insert(e)  // 传输完成后加入 GPU Queue       │
│                                                                │
│ optimal_latency = max(timeline_GPU, timeline_CPU, timeline_PCIe)│
└──────────────────────────────────────────────────────────────┘

┌── Multi-Stream Execution (GPU + CPU + PCIe 并行) ─────────────┐
│ Stream 0 (GPU compute):                                        │
│   for e in GPU_assigned_experts:                                │
│     // Marlin 4-bit quantized expert FFN                        │
│     h_4bit = quantize_4bit(h)  // activation quant              │
│     gate_out_4bit = MarlinGEMV(W_gate_4bit, h_4bit)            │
│     up_out_4bit = MarlinGEMV(W_up_4bit, h_4bit)                │
│     gate_act = SiLU(dequant(gate_out_4bit))                     │
│     fused = gate_act * dequant(up_out_4bit)                     │
│     out += gate_weight * dequant(MarlinGEMV(W_down_4bit,       │
│                                              quantize(fused)))  │
│                                                                │
│ Stream 1 (PCIe transfer):                                       │
│   for e in pcie_transfer_experts:                               │
│     // 在独立 stream 上执行，与 GPU compute overlap              │
│     cudaMemcpyAsync(                                            │
│         CPU_pinned_memory[e] → GPU_buffer[e],                   │
│         expert_size_bytes,  // ~300MB/expert (FP16)             │
│         cudaMemcpyHostToDevice,                                 │
│         stream_pcie                                             │
│     )                                                            │
│     cudaEventRecord(transfer_done_event, stream_pcie)           │
│     cudaStreamWaitEvent(stream_compute, transfer_done_event)    │
│                                                                │
│ CPU Thread Pool (CPU compute):                                  │
│   for e in CPU_assigned_experts:                                │
│     // llama.cpp C++ kernel                                     │
│     // Expert FFN: GEMM → SiLU → Hadamard → GEMM                │
│     cpu_gate_out = matmul_cpu(h, W_gate_cpu[e])                 │
│     cpu_up_out = matmul_cpu(h, W_up_cpu[e])                     │
│     cpu_fused = SiLU(cpu_gate_out) * cpu_up_out                 │
│     cpu_out = matmul_cpu(cpu_fused, W_down_cpu[e])              │
│     // 通过 pinned memory 写回 GPU                                │
│     cudaMemcpyAsync(cpu_out → GPU_output_buffer,                │
│                     output_size, cudaMemcpyHostToDevice)         │
│                                                                │
│ // 同步: CPU outputs 写回 GPU 后，GPU stream 聚合                │
│ cudaStreamSynchronize(stream_compute)                            │
│ final_out = weighted_sum(all_expert_outputs)                     │
└──────────────────────────────────────────────────────────────┘
```

  - **评估原理**：
    1. **TTFT/TBT 测量**：PyTorch CUDA event 计时，prefill 和 decode 分别测量。不同 input lengths（32/128/512/1024 tokens）和 GPU cache ratios（25%/50%/75%）的全面对比。
    2. **Ablation 分析**：分别启用/禁用 Scheduling、Prefetching、Caching 三个组件，测量各组件对 prefill/decode 延迟的独立贡献。
    3. **Cache Hit Rate 评估**：对比 MRS vs LRU 在不同 cache capacity（25%-75%）下的 expert cache hit rate。
    4. **Datasets**：MT Bench, Vicuna Bench, ChatGPT Prompts — prefill 评估用不同长度 traces，decode 评估用 ChatGPT Prompts。

  - **组件级加速贡献（Qwen2, 25% cache ratio）**：
    | Stage | Component | Speedup |
    |-------|-----------|---------|
    | Prefill | Scheduling | 1.26× |
    | Prefill | Prefetching | 1.06× |
    | Prefill | All | 1.31× |
    | Decode | Scheduling | 1.46× |
    | Decode | Prefetching | 1.15× |
    | Decode | Caching | 1.38× |
    | Decode | All | 1.86× |
    
    调度（Scheduling）在 prefill 和 decode 阶段均为最大单一贡献者；缓存（Caching）对 decode 贡献更显著（因 decode 阶段 expert 重用更频繁）。

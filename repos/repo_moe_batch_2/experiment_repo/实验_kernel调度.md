## Lancet: Accelerating Mixture-of-Experts Training via Whole Graph Computation-Communication Overlapping

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - Lancet 在 kernel 调度/运行时计算层面的实现：
    1. **Irregular All-to-All (All-to-Allv) 实现**：由于 Lancet 的 partition 方案沿 batch 维度分区导致每个 partition 发送给各 expert 的 token 数不均匀，实现了不规则形状的 all-to-all。采用双趟 NCCL 通信：第一趟 All-to-All 交换各 GPU 间实际要传输的 data size，第二趟 All-to-All 按照已知 size 传输实际数据。通过 grouped NCCL Send/Recv primitives 组合实现（非直接调用 ncclAllToAll）。
    2. **Tutel MoE Dispatching Kernel**：MoE 的 token-to-expert dispatch/gather 操作基于 Tutel 的 fast dispatching kernel 实现，处理动态 routing 后的 token 重排。
    3. **Pipeline Scheduling 内核**：partitioned computation/communication 被组织成 stage-based pipeline。相同类型的指令（all computation 或 all communication）组成一个 stage，各 partition 按 partition index 顺序执行。每个 kernel 的 start time 由依赖关系（数据依赖 + pipeline 顺序依赖）决定。
    4. **Caching Op Profiler + Communication Cost Model**：对每个 (partitioned) computation kernel 在不同 shape 下 profile 并缓存执行时间；通信 cost model 通过在不同 message size (1KB, 2KB, ..., max) 下 profile NCCL 通信操作构建，cost 间线性插值；对不规则 all-to-all，使用 static-shape approximation（以 C/n 容量的 uniform-shaped profile 值近似 n-partition 不规则 all-to-all 的通信时间）。
  - 实验比较：Lancet vs DeepSpeed vs Tutel vs RAF 在 V100/A100 集群上的训练吞吐量、iteration time decomposition（Non-overlapped Computation/Communication + Overlapped 的时间分解）

- 后端平台是什么，配置是什么。
  - **A100 GPU** (p4de.24xlarge): NVIDIA A100 80GB × 8 per node × 8 nodes, 4×100Gbps NIC per node
  - **V100 GPU** (p3dn.24xlarge): NVIDIA V100 32GB × 8 per node × 8 nodes, 1×100Gbps NIC per node
  - CUDA 11.3, NCCL 2.12.12 (PXN enabled), Ubuntu 20.06 Docker
  - all-to-all 通信基于 NCCL Send/Recv primitives，未使用 ncclAllToAll 高层 API

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估脚本：`run_exp_configs.py`（GitHub 仓库提供）
    - `--lancet-profile`：profile 阶段，生成 execution time cache
    - `--lancet-opt`：应用 Lancet 优化并运行
    - 无 flag：baseline 运行（RAF 无 Lancet 优化）
  - 通信 profiling：`create_nccl_profiles.py` 生成 NCCL communication cost model
  - 修改/新增内容：
    1. **Irregular All-to-All Kernel**：13K LoC C++ 中的通信层实现，基于 NCCL `ncclSend`/`ncclRecv` 组合成 grouped communication。Input/Output buffer 固定 shape (G × C)，实际数据只填充 buffer 的部分区域（由 gating 结果决定），避免传输 padding token。
    2. **Tutel-based MoE Dispatch**：复用 Tutel 的 fast expert dispatch kernel 进行 token permutation
    3. **Partition 约束函数 F_Z**：为所有 Transformer 计算算子（MatMul, LayerNorm, Attention, Activation 等）定义 partition axis 约束规则
    4. **Pipeline Scheduler**：新的 kernel launch 调度器，根据 partition index 和数据依赖编排所有 partitioned kernel/communication 的 launch 顺序

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - **开源**：https://github.com/hikettei/Lancet (Apache-2.0)，AWS Labs 镜像 https://github.com/awslabs/Lancet-Accelerating-MoE-Training-via-Whole-Graph-Computation-Communication-Overlapping
  - **Irregular All-to-All Kernel 执行原理**：
```
Input: Gating result -> per-expert token counts per GPU
  Step 1: 在每 GPU 上根据 gating function 决定每个 expert 收到的 token 数
          Input buffer (shape G×C) 仅填充实际 token
  Step 2: 第一趟 All-to-All (data size exchange)
          - 每 GPU 对每个 target GPU 发送实际要传输的 data size
          - 实现: grouped NCCL Send/Recv (每个 target/src GPU 一个 Send/Recv)
          - 此时仅传输 int 类型的 count 信息，通信量极小
  Step 3: 第二趟 All-to-All (actual data transfer)
          - 每 GPU 根据 Step 2 获知的 size 发送/接收实际 token data
          - 实现: grouped NCCL Send/Recv with known sizes
          - 由于不传输 padding tokens，总体通信量低于 uniform-shaped all-to-all
  Step 4: Expert computation
          - 各 GPU 上每个 expert 处理收到的 Ci 个 token (Ci ≤ C)
          - Tutel kernel 执行 expert FFN (可能含 padding 到 block size)
  Step 5: Reverse irregular all-to-all (同上 Steps 1-3 的逆过程)
          - 将 expert output 发送回原 GPU
```
  - **Pipeline 执行全过程**（前向传播中 3 个 partition 的例子）：
```
Timeline:
  Partition 0:  [Non-MoE Compute_0] [All-to-All_0] [Expert_0] [Non-MoE Compute_0_post]
  Partition 1:     [Non-MoE Comp_1] [All-to-All_1] [Expert_1] [Non-MoE Compute_1_post]
  Partition 2:        [Non-MoE Comp_2] [All-to-All_2] [Expert_2] [Non-MoE Compute_2_post]

  其中 Non-MoE Compute_i 与 All-to-All_{i-1} 和 Expert_{i-1} 重叠执行
  All-to-All_i 与 Non-MoE Compute_{i-1}_post 和 Expert_{i-1} 重叠执行
  Stage 结构：
    Stage 0 (Compute):  [NMC_0, NMC_1, NMC_2] 按 partition index 顺序 launch
    Stage 1 (Comm):     [A2A_0, A2A_1, A2A_2]
    Stage 2 (Compute):  [Expert_0, Expert_1, Expert_2]
    Stage 3 (Compute):  [Post_0, Post_1, Post_2]
  Pipeline Scheduler 计算每个 kernel 的 start_time：
    start(NMC_i) = max(end(NMC_{i-1}), end(Post_{i-1} of prev layer))
    start(A2A_i) = max(end(NMC_i), end(A2A_{i-1}))
    start(Expert_i) = max(end(A2A_i), end(Expert_{i-1}))
```

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

## Hecate: Unlocking Efficient Sparse Model Training via Fully Sharded Sparse Data Parallelism

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - Hecate 在 kernel 调度层面的核心实现：
    1. **Sparse Collective Communication Scheduling**：两个新稀疏通信原语 SparseAllGather 和 SparseReduceScatter 的 NCCL 实现与调度。利用 NCCL group calls 同时调度一系列 Broadcast（SparseAllGather）和 Reduce（SparseReduceScatter）操作，每个 Broadcast/Reduce 操作对应一个 chunk（即一个 expert）到一组 target devices 的通信。
    2. **Communication-Computation Overlap Scheduling**：将 SparseAllGather 和 SparseReduceScatter 的通信与 Attention layer 的前向/后向计算重叠。Forward pass 中，SparseAllGather 的延迟 ≤ Attention forward 时间即可完全隐藏；Backward pass 中，SparseReduceScatter（当前层梯度 reduce）+ SparseAllGather（下一层 re-materialize）同时重叠于 Attention backward（后者耗时约为 forward 2×）。
    3. **Topology-Aware Dispatching**：Dispatcher 在 token dispatching（All-to-All）阶段优先 intra-node 通信，仅当 source node 内无 expert replica 时才跨 node dispatching。均匀分配 tokens 到同一 expert 的多个 replica devices。
    4. **Calibration Stage**：在 MoE gate 输出真实 token assignment 后，重新运行 Algorithm 1，用实际负载和剩余 memory capacity 判断是否追加一次 on-critical-path 的 SparseAllGather 来进一步减少 imbalance。
    5. **Re-materialization Scheduling**：Hecate-RM 在 backward pass 释放已用 expert 参数后，重新调度 SparseAllGather 来物化下一层的 expert 参数，形成 "release → re-materialize" 的流水线。
  - 实验比较：(1) FSSDP vs EP 的 layer-wise speedup（2.8-18.8×，geo-mean 11.87×）；(2) All-to-All 通信时间对比（Hecate 减少 12.3× vs EP）；(3) 各系统 critical path 分解（FasterMoE 的 FusedKernel、FlexMoE 的 Rearr overhead、Hecate 的 SpAG/SpRS overhead）；(4) Hecate-RM re-materialization overhead（3.6× 增加 sparse collectives 通信但仍优于 baseline 1.4×）。

- 后端平台是什么，配置是什么。
  - **Cluster A**：4× AWS p3dn.24xlarge nodes，每 node 8× NVIDIA V100-32G GPU（NVLink 300 GB/s intra-node），node 间 100 Gbps 网络。
  - **Cluster B**：4× AWS p4d.24xlarge nodes，每 node 8× NVIDIA A100-40G GPU（NVSwitch 600 GB/s intra-node），node 间 400 Gbps 网络。
  - 在网络带宽较低的 Cluster A (100 Gbps) 上 All-to-All straggler 效应更显著，Hecate 加速效果更明显。

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 **PyTorch** + **NCCL** 实现。使用 **Megatron-LM** 作为训练框架，baseline systems 仅优化 MoE layer 训练。
  - 修改的内容：
    - 在 NCCL 之上实现了 SparseAllGather 和 SparseReduceScatter 两个稀疏通信原语，通过 `ncclGroupStart/End` 包装一组 Broadcast（spAG）或 Reduce（spRS）操作。
    - 实现 Communicator 组件：维护通信任务队列，调度执行稀疏 collectives 和 token dispatching All-to-All。
    - 实现 Scheduler 组件：基于 expert load 分布估计（滑动窗口平均，w=5）和 overlap degree / memory capacity 约束，生成 placement plan 并驱动稀疏 collectives 调度。
    - 实现 Dispatcher 组件：拓扑感知的 token 路由决策。
    - Hecate 不实现 expert execution 与 All-to-All 的 overlap（认为正交），稀疏 collectives 仅与 Attention computation 重叠。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 论文未公开 Hecate 代码。作为 prototype system，稀疏 collectives 用 NCCL group calls 实现，作者指出更高效的稀疏 collective 算法（利用数据稀疏性和网络拓扑）留作 future work。
  - SparseAllGather 的 NCCL 实现原理与执行全过程：

```
┌── SparseAllGather 执行模型 ──────────────────────────────┐
│ 逻辑输入：expert parameters 划分为 equal-sized chunks     │
│ C = {C_0, C_1, ...}（每个 chunk = 一个 expert 的参数）   │
│ Pre-condition P_0: 每个 chunk 唯一归属于某 source device │
│ Post-condition P_1: P_0 ⊆ P_1（物化目标 placement）      │
│                                                           │
│ NCCL 实现：                                               │
│   ncclGroupStart()                                        │
│   for each (c, d_target) in P_1 \ P_0:                   │
│       // 对每个需要物化的 (expert, target_device) 对      │
│       d_src = 唯一持有 chunk c 的 device (from P_0)       │
│       ncclBroadcast(chunk_c_data,                        │
│                      root=d_src,                          │
│                      comm=sub_group_containing_d_target)  │
│   ncclGroupEnd()                                          │
│                                                           │
│ 通信量分析：                                              │
│ - expert 参数大小 = expert_size bytes                     │
│ - 需物化的 expert set Ĉ（|Ĉ| ≤ |C|）                      │
│ - 稀疏度 λ = |Ĉ| / |C|                                   │
│ - 每个物化 expert 以 Broadcast 发送到 target devices     │
│ - 最坏情况：某 device 需接收所有 Ĉ 中的 chunks            │
│ - 通信量上界：O(λ · S)，其中 S = |C| × expert_size      │
│ - 相比 FSDP AllGather 的 O(S)，当 λ << 1 时显著降低     │
└───────────────────────────────────────────────────────────┘

┌── SparseReduceScatter 执行模型 ──────────────────────────┐
│ Pre-condition P_0: gradients 分布在多个 device 上        │
│ Post-condition P_1: 每个 chunk 的 reduce 结果在唯一      │
│                      source device (P_1 surjective)      │
│                                                           │
│ NCCL 实现：                                               │
│   ncclGroupStart()                                        │
│   for each (c, d_src) in P_1:                            │
│       // 对每个需 reduce 到 source 的 chunk              │
│       ncclReduce(chunk_c_grad_data,                      │
│                   root=d_src,                             │
│                   comm=sub_group_with_replica_of_c)       │
│   ncclGroupEnd()                                          │
│                                                           │
│ 与 spAG 对称：spRS(P', P) 的通信量上界 = O(λS)          │
│ 与 rearrangement 系统 AllReduce 等价：                   │
│   Vol(AllReduces) ≈ Σ_i 2(|D_i|-1)/|D_i| · S/|C|        │
│   ≈ O(2λS) ≈ Vol(spAG) + Vol(spRS)                      │
└───────────────────────────────────────────────────────────┘

┌── 通信-计算重叠调度时序 ─────────────────────────────────┐
│ Forward:                                                  │
│   [Attention Forward]                                     │
│   ├── SparseAllGather (overlap with Attn Fwd) ──┤        │
│   [MoE Gate + Token Dispatch + Expert Comp]               │
│                                                           │
│ Backward:                                                 │
│   [Attention Backward]  ← 耗时约 2× Forward               │
│   ├── SparseReduceScatter (layer l gradients) ──┤        │
│   ├── SparseAllGather (layer l+1 re-materialize) ┤        │
│   [MoE Gate Bwd + Expert Bwd]                             │
│                                                           │
│ 约束条件：                                                │
│ - t = T_non-MoE · bw / expert_size                        │
│   (overlap degree: 可在 attention 时间内隐藏通信的        │
│    最大 expert 数)                                        │
│ - 拓扑感知：bw 异构时使用 inter-node bandwidth            │
│             同构时使用 uniform inter-device bandwidth     │
└───────────────────────────────────────────────────────────┘
```

## HEXA-MoE: Efficient and Heterogeneous-aware MoE Acceleration with ZERO Computation Redundancy

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - HEXA-MoE 包含三个 kernel 调度/运行时计算层面的实现：
    1. **Re-Index Vector based Expert-Specific CUDA Kernels**：实现 ESMM、ESS、ESTMM 三个 expert-specific 算子的 GPU kernel。通过构建 re-index vector（按 routing choice 重排 token indices，同 expert 的 token 聚集为 sub-vector，padding -1 对齐 tiling size）作为 I/O 指导。ESMM kernel：thread-block 加载 sub-vector → 按 vector 值加载 tokens 和对应 expert 权重（同一 expert 的 tokens 只需加载一次权重）→ 沿 input feature 维度累加 dot product → 按 sub-vector 写回 HBM。ESS kernel：每个 thread-block 分配某 expert 的某些 channel → 加载该 expert 对应 sub-vector 中 tokens → 累加后写回 HBM。ESTMM kernel：输入以 re-indexed 格式提供，两输入共享 re-index vector → thread-block 加载同 expert 两输入特定 channel → 累积外积 → 写回 HBM。
    2. **Expert-Specific Fused Kernel (ESFK)**：将 backward pass 中的 ESS、ESTMM、ESMM 融合为单一 kernel。通过统一各算子的 thread-block shape 为 (WARP, TIMES)，并将 thread-grid 扩展为 3 维使各算子 grid 可对齐并聚合。单 MoE 层 backward 仅需 2 个 fused kernels + 1 个 element-wise dot product。
    3. **Pipeline-Shared Cache + Communication-Computation Overlap**：在 data-centric 配置下，每设备分配额外 HBM 区域作为 pipeline-shared cache，动态缓存 all gather 来的 MoE parameter shards。All gather 通信与 attention/router 计算重叠，backward pass 无需保存完整 MoE 参数（通过 cache 动态获取），解决 Janus 等方法的 backward 内存膨胀问题。
  - 实验比较：(1) Memory Analysis: HEXA-MoE vs Tutel vs MegaBlocks 的 GPU 内存占用，2 homogeneous GPUs, 8 global experts, top-1~top-8 routing；(2) Latency Analysis: 平均每步训练延迟，4 homogeneous GPUs, 4 experts, 不同 batch size；(3) Ablation: 各组件的 memory footprint breakdown 和 latency breakdown。

- 后端平台是什么，配置是什么。
  - 同构机器 M_homo：4× NVIDIA GeForce RTX 4090 (24 GB)，CPU 2× Intel Xeon Platinum 8352V 2.10GHz。
  - 异构机器 M_hete：1× NVIDIA TITAN RTX (24 GB) + 1× NVIDIA GeForce RTX 2080 Ti (11 GB)，CPU 2× Intel Xeon Gold 6130 2.10GHz。
  - CUDA kernel 参数：BLK (block size)，WARP (warp size=32)，TIMES (thread block 有 WARP×TIMES 个线程)。使用 nvcuda::wmma 接口调用 Tensor Core 做 16×16×16 矩阵乘法。

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 PyTorch + CUDA C++ 实现 expert-specific kernels（ESMM、ESS、ESTMM、ESFK）。
  - 修改 NCCL all gather / all reduce 通信后端，适配 tensor parallelism 替代 expert parallelism。
  - 使用 PyTorch automatic mixed precision 训练。
  - 评估指标：NVIDIA SMI 监控 GPU 内存占用 (GB)，PyTorch CUDA event 计时测量每训练步延迟 (s)，2k steps 取平均。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源链接：https://github.com/UNITES-Lab/HEXA-MoE（支持 Triton 和 CUDA 两种 kernel 实现）
  - ESMM Kernel 执行原理全过程（基于 Algorithm 2）：

```
┌── Kernel Input ────────────────────────────────────────┐
│ R: routing choice [N], v: re-index vector [N']          │
│ x: input tokens [N, D1], w: weights [E, D1, D2]         │
│ b: bias [E, D2]                                         │
│ BLK: tiling size, WARP=32, TIMES: threads/BLK           │
└────────────────────────────────────────────────────────┘

┌── ESMM CUDA Kernel ──────────────────────────────────┐
│ Parallel for i in range(0, N', BLK):                   │
│   Parallel for j in range(0, D2, BLK):                 │
│     exp = R[v[i]]                    // expert index   │
│     c = b[exp, j:j+BLK].repeat(BLK, 1)  // init bias  │
│     for k in range(0, D1, BLK):                        │
│       // Load BLK tokens for this expert               │
│       Parallel for t = 0 to BLK-1:                     │
│         if v[i+t] != -1:                               │
│           xsub[t] = x[v[i+t], k:k+BLK]                 │
│         else: xsub[t] = 0       // skip padding        │
│       wsub = w[exp, k:k+BLK, j:j+BLK]                  │
│       c += xsub @ wsub           // Tensor Core MMA    │
│     // Write back                                      │
│     Parallel for t = 0 to BLK-1:                       │
│       if v[i+t] != -1:                                 │
│         y[v[i+t], j:j+BLK] = c[t]                      │
└────────────────────────────────────────────────────────┘
```

  - Re-Index Vector 构建原理（Algorithm 1）：
    1. 统计每个 expert 的 token 数量 ctr[e]（atomicAdd）
    2. 将 ctr[e] 向上取整到 BLK 的倍数
    3. 计算累积偏移 idx[e]（prefix sum）
    4. 按 routing choice 将 token index 写入 v[idx[R[i]]++] = i
    5. v 中未填满的 BLK 位置填充 -1

  - ESFK 融合原理（Table 6）：
    所有算子 thread-block shape 统一为 (WARP, TIMES)，thread-grid 扩展为 3 维。ESS grid: (E, D2/(TIMES·BLK), 1) → 扩展第三维为 1；ESMM grid: (N'/BLK, D1/(TIMES·BLK), 1) → 扩展第三维为 1；ESTMM grid: (E, D1/(TIMES·BLK), D2/(TIMES·BLK))。聚合后 ESFK grid 第三维 = N'/BLK + D2/BLK + D2/(TIMES·BLK)。

  - Pipeline-Shared Cache 原理：
    ```
    # Data-centric 配置下的内存管理
    # 每设备在 HBM 额外分配 cache 区域
    # Forward: all gather MoE shards → 写入 cache → ESMM 计算
    # Backward: 从 cache 读取所需的 gathered shards（无需永久保存）
    # All gather 与 attention/router 在分离 CUDA stream 上 overlap
    ```

## FloE: On-the-Fly MoE Inference on Memory-constrained GPU

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - FloE 包含两个 kernel 调度/运行时计算层面的实现：
    1. **Efficient Sparse GEMV Kernel（Section 3.4.1）**：基于 Triton（参考 CATS kernel）实现的自定义稀疏 GEMV kernel。将 W_down 转置为列主序存储（W_down^T），与 gate projection 的列对齐；根据稀疏掩码选择性加载 W_gate 和 W_down^T 的列，减少内存访问次数。将 SiLU 激活和 element-wise 乘法融合到每个 block 计算中，节省中间结果 x' 的多次存储/加载，减少 kernel launch 次数。在 RTX 3090 上，90% 稀疏度时达接近 2× 加速。
    2. **Compact Asynchronous Transfer（Section 3.4.2）**：紧凑权重布局（co-locate gate 列 + down 行到连续 DRAM 区域，chunk 大小翻倍）；CPU 端 AVX-512 SIMD 指令 + 多线程打包压缩权重到 pinned memory；跨多 CUDA stream 异步发送传输请求，最大化 PCIe 带宽利用率。对比 PyTorch 原生实现加速 12.6×，达到 PCIe 4.0 峰值带宽的 88%。
  - 实验比较：单 expert sparse GEMV kernel 延迟 vs dense baseline（sparsity=0），在 H100/A100/A6000/RTX 3090 上对比不同稀疏度（50%/60%/70%/80%/90%）的加速比。传输效率测试：对比不同 chunk size（1~200）下 compact async transfer vs PyTorch 原生的传输延迟和带宽利用率。

- 后端平台是什么，配置是什么。
  - Single-expert kernel 测试：H100（计算吞吐高，但 kernel launch overhead 限制稀疏加速），A100，A6000，GeForce RTX 3090（consumer-grade GPU，稀疏加速最明显）。
  - 传输测试：RTX 3090 + 64核 CPU + 256GB DRAM + PCIe 4.0 ×16。

- 评估性能的软件/脚本是什么。修改了什么。
  - 使用 Triton（Tillet et al., 2019）实现 sparse GEMV kernel，修改自 CATS kernel。
  - 使用 PyTorch 自定义 C++ 扩展实现 compact async transfer（AVX-512 + pinned memory + multi-stream）。
  - 单 expert 延迟测试：C4 数据集 500 tokens，80 warmup + 200 timed iterations。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 论文未明确给出 FloE 开源链接。sparse GEMV kernel 基于 Triton 构建，可参考 Triton（https://github.com/triton-lang/triton）和 CATS（https://openreview.net/forum?id=v3w2a7EInO）。
  - Sparse GEMV kernel 评估原理：
    1. **输入**：hidden state x (1×4096)，sparse threshold t_ij，expert weights {W_gate (4096×14336), W_down^T (4096×14336, 列主序转置), W_up (4096×14336)}。
    2. **Kernel 执行流程**：x 与 W_up 全精度 GEMV → 产生激活向量 v (1×14336) → |v| 与 t_ij 比较生成 binary mask → GPU 根据 mask 选择性从 global memory 加载 W_gate 和 W_down^T 被选中的列（列宽 = d_hidden=4096）→ Triton block 内融合执行 SiLU + element-wise multiply + sparse GEMV → 输出结果 y (1×14336)。
    3. **性能测量原理**：使用 CUDA event 计时，warmup 80 次后 200 次 timed iteration 取平均。对比 dense baseline（sparsity=0，全量加载全量计算）的 wall-clock 延迟（ms）。利用 Nsight Systems 验证 PCIe 带宽利用率和 kernel 时间线。
    4. **传输效率评估原理**：随机选取 20% expert 权重列，从 DRAM 通过不同 chunk size 传输到 VRAM。每个 chunk 对应一个 CPU 线程 + pinned memory 打包 + CUDA stream 异步拷贝。使用 PyTorch CUDA event 和 CPU timer 测量传输延迟，计算实际带宽 = transferred_bytes / latency，除以 PCIe 4.0 峰值理论带宽（~25GB/s 实测上限）得到利用率。

## Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - Fiddler 包含两个 kernel 调度/运行时计算层面的实现：
    1. **CPU AVX512_BF16 Expert 计算 kernel**：利用 Intel AVX512_BF16 指令集实现的自定义 CPU expert FFN 计算 kernel。PyTorch 原生不支持 BF16 的 AVX512 指令，Fiddler 手动实现以提升 CPU 端 expert 计算吞吐。
    2. **异构后端运行时调度（Algorithm 1）**：在 CPU 和 GPU 两种后端间动态调度 expert 计算。基于 latency model（GPU 延迟恒定，CPU 延迟随输入量线性增长）和输入 token 数量 s，在每个 MoE 层运行时决定每个 expert 的执行后端——GPU 直接执行、GPU+PCIe weight transfer 执行、或 CPU 执行（activation copy + CPU compute + output copy back）。
  - 实验比较：
    - 微基准：测量 weight copy (CPU→GPU)、activation copy (GPU→CPU)、GPU expert execution (不同 input size)、CPU expert execution (不同 input size) 的延迟
    - 宏基准：Fiddler vs DeepSpeed-MII vs Mixtral-Offloading vs llama.cpp 在单 batch 推理、长 prefill、beam search 三种场景

- 后端平台是什么，配置是什么。
  - GPU 后端：NVIDIA Quadro RTX 6000 (24GB) / RTX 6000 Ada (48GB)
  - CPU 后端：Intel Xeon Gold 6126 (48 cores, Env1) / Intel Xeon Platinum 8480+ (112 cores, Env2)
  - CPU-GPU 传输：PCIe Gen3 x16 (32GB/s, Env1) / PCIe Gen4 x16 (64GB/s, Env2)
  - CPU 指令集：AVX512_BF16（Intel Xeon Platinum 8480+ 支持 AMX/AVX512_BF16）

- 评估性能的软件/脚本是什么。修改了什么。
  - Fiddler 基于 PyTorch 构建，自建 microbenchmark 脚本测量各操作延迟：
    - Weight copy latency (CPU→GPU)：每个 expert weight ~300MB，测量 32 层平均和标准差
    - Activation copy latency (GPU→CPU)：测量 32 层平均和标准差
    - GPU expert execution latency：分别在 input size N=1,2,4,8,16,32,64 下测量
    - CPU expert execution latency：分别在 input size N=1,2,4,8,16,32,64 下测量
  - 修改内容：
    - **新增 CPU AVX512_BF16 kernel**：替代 PyTorch 默认 CPU GEMM，针对 expert FFN 的 (input×4096)×(4096×14336) 矩阵乘优化
    - **运行时调度逻辑**：在 PyTorch forward path 插入 `cpu_lat(s)` / `gpu_lat(s)` / `trans_lat()` 决策逻辑
    - **Latency model 校准**：初始化阶段运行微基准测量三个函数所需的常数参数

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - **开源情况**：代码开源在 https://github.com/efeslab/fiddler
  - **CPU AVX512_BF16 Kernel 执行原理全过程**：

    ```
    ┌── Kernel Input ──────────────────────────────────────────┐
    │ activation: float32/bf16 tensor [s, 4096]  (GPU→CPU copy) │
    │ expert_weights: {                                         │
    │   W_gate: [4096, 14336]  // gate projection               │
    │   W_up:   [4096, 14336]  // up projection                 │
    │   W_down: [14336, 4096]  // down projection               │
    │ }  // 常驻 CPU pinned memory, 16-bit precision            │
    │ s: number of input tokens (1 for single-batch decode,     │
    │    up to thousands for prefill)                            │
    └──────────────────────────────────────────────────────────┘

    ┌── CPU AVX512_BF16 Expert FFN Kernel ──────────────────┐
    │ // 利用 AVX512_BF16 VDPBF16PS 指令 (每周期 32 个 BF16 MAC) │
    │ // PyTorch 默认使用 FP32 GEMM, 无法利用 BF16 硬件加速      │
    │                                                           │
    │ // Step 1: gate projection                                │
    │ gate_out = matmul_avx512_bf16(activation, W_gate)         │
    │ // [s, 4096] × [4096, 14336] → [s, 14336]                │
    │                                                           │
    │ // Step 2: up projection                                  │
    │ up_out = matmul_avx512_bf16(activation, W_up)             │
    │ // [s, 4096] × [4096, 14336] → [s, 14336]                │
    │                                                           │
    │ // Step 3: SiLU activation                                │
    │ gate_act = SiLU(gate_out)  // element-wise                │
    │ // SiLU(x) = x * sigmoid(x)                               │
    │                                                           │
    │ // Step 4: gated fusion                                   │
    │ fused = gate_act * up_out  // element-wise multiply       │
    │                                                           │
    │ // Step 5: down projection                                │
    │ output = matmul_avx512_bf16(fused, W_down)                │
    │ // [s, 14336] × [14336, 4096] → [s, 4096]                │
    │                                                           │
    │ // 关键：每个 matmul 内部使用 AVX512 tile 分块：            │
    │ // - 每次加载 32 个 BF16 元素到 ZMM 寄存器                  │
    │ // - VDPBF16PS 指令计算 32 个 BF16 点积                    │
    │ // - 累加结果到 FP32 accumulator                          │
    │ // - Tile 大小选择最小化 CPU cache miss                    │
    └──────────────────────────────────────────────────────────┘

    ┌── Runtime 调度决策 (Algorithm 1) ─────────────────────┐
    │ for each expert j in layer l:                             │
    │   s = inp_size[j]  // #tokens routed to expert j          │
    │   if s == 0: skip                                         │
    │   if is_at_gpu(l, j):                                     │
    │     // Strategy (a): 纯 GPU 执行                          │
    │     output = cuda_expert_ffn(activation_gpu, W_gpu)       │
    │   elif cpu_lat(s) > gpu_lat(s) + trans_lat():             │
    │     // Strategy (b): GPU+CPU→GPU weight transfer          │
    │     W_gpu = cudaMemcpyAsync(W_cpu → W_gpu, PCIe)         │
    │     output = cuda_expert_ffn(activation_gpu, W_gpu)       │
    │   else:                                                   │
    │     // Strategy (c): CPU execution                        │
    │     act_cpu = cudaMemcpyAsync(act_gpu → act_cpu, PCIe)   │
    │     output = avx512_bf16_expert_ffn(act_cpu, W_cpu)      │
    │     cudaMemcpyAsync(output_cpu → output_gpu, PCIe)        │
    └──────────────────────────────────────────────────────────┘
    ```

    **微基准数据（Figure 7, Appendix A）**：
    - Weight copy: GPU computation 的 2-5× 时间（主要开销）
    - GPU execution: 基本恒定于 batch size（Env1 batch=1 时因 PyTorch 单 batch 使用不同实现有约 10% 差异）
    - CPU execution: 随 input size 线性增长
    - Activation copy: <1% of CPU single-input latency（可忽略）
    - 建模简化：gpu_lat(s) = constant, cpu_lat(s) ∝ s, activation transfer latency ≈ 0

    **策略 (b) vs (c) 的 trade-off 分析**：
    | Input size s | Strategy (b) 延迟 | Strategy (c) 延迟 | 最优 |
    |-------------|-------------------|-------------------|------|
    | s=1 (decode) | trans_lat + gpu_const | cpu_const × 1 | (c) CPU |
    | s=256 (prefill) | trans_lat + gpu_const | cpu_const × 256 | (b) GPU+transfer |
    | 阈值 s_threshold | = trans_lat / cpu_slope | — | 切换点 |

## FUSCO: High-Performance Distributed Data Shuffling via Transformation-Communication Fusion

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - FUSCO 的核心 kernel 调度组件是 **Data-Fused Communication Engine (dComm)**，约 2000 行 C++/CUDA 实现。dComm 通过以下机制实现 runtime 级别的 fused data+communication：
    1. **Segment Descriptor 驱动的 Gather/Scatter Kernel**：发送端 GPU kernel 读取 descriptor 数组（连续存放的 {addr, size} 对），根据累计已传输字节数定位当前 segment，从非连续内存中 gather 数据到 NIC ring buffer，在此过程中完成 layout transformation（将 expert-major layout 转换为 device-major layout），无需额外的 permute kernel。
    2. **Pipelined Slice 传输**：将多个 logical segments 打包为 slice（远大于单个 segment），GPU producer kernel 将 slice 写入 ring buffer，NIC consumer（RDMA）从 ring buffer 读取并发送。由于 RDMA 传输时间通常超过 GPU slice 准备时间，GPU memory copy 和 NIC 传输完全重叠。
    3. **GPUDirect P2P 节点内 Kernel**：对于 intra-node 传输，dComm 使用 GPUDirect P2P 直接 GPU-to-GPU copy，在 copy 路径中集成 descriptor 解释逻辑，inline 完成 layout transformation。
  - 实验比较 FUSCO 与 NCCL（使用 `index_select` 等 PyTorch 算子做显式重排+通信）和 DeepEP（基于 NVSHMEM 的 warp-specialized kernel + IBGDA）在三种流量模式下的通信微基准性能，以及 per-component 消融实验。

- 后端平台是什么，配置是什么。
  - GPU：NVIDIA H100 80GB HBM3（每节点 8 张，共 64 张）
  - 节点内互联：NVLink，每 GPU 18 条 link，理论聚合带宽约 480 GB/s per GPU
  - 节点间互联：Mellanox ConnectX-7 400 Gbps NIC × 10 per node（RoCE），理论跨节点带宽约 50 GB/s
  - CUDA 12.9，NCCL 2.26.3
  - 通信微基准参数：EP=64，hidden_dim=7168，top-k=8，num_experts=256（与 DeepSeek-V3 一致）

- 评估性能的软件/脚本是什么。修改了什么。
  - **通信微基准测试**：自建 benchmark 将 MoE 通信分为三阶段测量——preprocessing（路由结果转换为通信调度）、rearrangement（token 重排以对齐通信或 expert layout）、communication（all-to-all dispatch + combine）。
  - **宏基准测试**：
    - 训练：Megatron-LM，per-iteration training time
    - 推理：SGLang + prefill-decode disaggregation，time-to-first-token (TTFT)
  - **消融实验**：分别禁用 dComm（回退到 NCCL + 显式重排）、Planner（回退到默认 all-to-all，每个 token 独立发送）、Balancer（回退到同 index 的静态分组），测量性能退化。
  - **修改内容**：FUSCO 的 dComm runtime 作为独立 collective primitive（类似 send/recv/allgather）暴露，通过扩展的 PyTorch distributed backend 调用。Communication Planner 和 Online Balancer 约 1000 行 Python，使用 PyTorch GPU operators（sum, argsort, gather, scatter）构建 descriptor。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - **开源情况**：论文声明 "Our code and data will be made publicly available"，截至分析时未找到公开代码仓库。FUSCO 基于 NCCL（https://developer.nvidia.com/nccl）和 PyTorch（https://pytorch.org）构建。
  - **dComm Kernel 执行原理全过程**：
    ```
    ┌── Kernel Input ──────────────────────────────────────┐
    │ descriptor_list = [                                    │
    │   {addr: 0x7f00, size: 8192},   // token₀ 在 GPU mem │
    │   {addr: 0x8a00, size: 8192},   // token₁              │
    │   {addr: 0x9100, size: 14336},  // token₂ (大 token)   │
    │   ...                                                 │
    │ ]  // 连续存放在 GPU global memory                      │
    │ ring_buffer[NIC_RING_SIZE]  // NIC 可见的环形缓冲区     │
    │ total_bytes_to_send = sum(descriptor_list[i].size)     │
    └───────────────────────────────────────────────────────┘
    
    ┌── GPU Producer Kernel (每个 slice 一次 launch) ──────┐
    │ slice_size = max_slice_bytes (如 1MB)                  │
    │ slice_start = slice_id * slice_size                    │
    │                                                        │
    │ // 定位当前 segment                                    │
    │ cumsum = 0                                             │
    │ for desc in descriptor_list:                           │
    │     if cumsum + desc.size > slice_start:               │
    │         // 当前 segment 跨越 slice 边界                │
    │         offset_in_seg = slice_start - cumsum           │
    │         bytes_to_copy = min(                           │
    │             desc.size - offset_in_seg,                 │
    │             slice_size - bytes_copied                  │
    │         )                                              │
    │         cudaMemcpyAsync(                               │
    │             ring_buffer + bytes_copied,                │
    │             desc.addr + offset_in_seg,                 │
    │             bytes_to_copy,                             │
    │             cudaMemcpyDeviceToDevice                   │
    │         )                                              │
    │         // ↑ 此 copy 完成了 layout transformation:     │
    │         //   从 expert-major 非连续布局 gather 到       │
    │         //   连续的 device-major ring buffer            │
    │         bytes_copied += bytes_to_copy                  │
    │         if bytes_copied >= slice_size: break           │
    │     cumsum += desc.size                                │
    │                                                        │
    │ // Signal NIC: slice 已就绪                             │
    │ __threadfence_system()                                │
    │ *slice_ready_flag = 1                                  │
    └───────────────────────────────────────────────────────┘
              │
              │ NIC 读 ring_buffer，RDMA Write 到远端 GPU
              │ (GPU 继续处理下一个 slice)
              ▼
    ┌── Receiver GPU Kernel ───────────────────────────────┐
    │ // 镜像逻辑：接收端 descriptor 数组指定 scatter 目标    │
    │ for desc in receiver_descriptor_list:                 │
    │     // desc.addr = 该 segment 应在 expert activation   │
    │     //              tensor 中的最终位置                │
    │     cudaMemcpyAsync(                                   │
    │         desc.addr,  // 直接写入 expert 计算所需的 layout│
    │         recv_buffer + bytes_received,                 │
    │         desc.size,                                     │
    │         cudaMemcpyDeviceToDevice                       │
    │     )                                                  │
    │     bytes_received += desc.size                        │
    └───────────────────────────────────────────────────────┘
    ```

    **NCCL Baseline 的 kernel 执行过程（对比）**：
    ```
    // Baseline (NCCL + PyTorch): 3 步，5 次 memory pass
    // Step 1: 显式重排 (2 memory passes: read + write)
    permuted = torch.index_select(tokens, dim=0, index=rank_indices)  
    
    // Step 2: All-to-all 通信 (implicit memory pass via NCCL)
    exchanged = nccl_all_to_all(permuted)
    
    // Step 3: 再次重排 (2 memory passes: read + write)
    expert_input = torch.index_select(exchanged, dim=0, index=expert_indices)
    
    // 总计: 5 次 memory pass（含重排）+ 1 次网络通信
    // FUSCO: 1 次 memory pass（gather→ring buffer）+ 1 次网络通信（pipelined）
    ```

    **Pipelined 时序图**：
    ```
    Time →
    GPU: |== Slice₀ Gather ==|== Slice₁ Gather ==|== Slice₂ Gather ==|
    NIC:                      |== RDMA Slice₀ ===|== RDMA Slice₁ ===|
    ```
    GPU gather 时间 < NIC RDMA 时间，因此 GPU 操作完全隐藏在通信延迟中。

    **三种流量模式下的 kernel 性能数据**（16k seqlen，单位 ms，FUSCO vs NCCL vs DeepEP）：
    - Real-world traffic: 86.84 vs 144.30 vs 119.48（1.66× / 1.38× speedup）
    - Single-node routed: 40.99 vs 157.28 vs 82.17（3.84× / 2.01× speedup）
    - Load-imbalanced: 151.30 vs 338.99 vs 213.74（2.24× / 1.42× speedup）

    **消融实验**（Table 3，16k seqlen，real-world traffic，单位 ms）：
    | Configuration | Latency | Degradation |
    |--------------|---------|-------------|
    | FUSCO (full) | 86.84 | - |
    | dComm-off | 119.48 | -27.32% |
    | Planner-off | 124.35 | -30.17% |
    | Balancer-off | 95.13 | -8.72% |

## HeterMoE: Efficient Training of Mixture-of-Experts Models on Heterogeneous GPUs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - HeterMoE 在 kernel 调度/运行时计算层面的核心实现：
    1. **Zebra Parallelism (ZP) 调度**：替代传统 Expert Parallelism (EP)。在一个 ZP group 内，expert 模块分布在 N 个 expert GPU（older generation），attention 等其余模块复制在 M 个 attention GPU（newer generation）。ZP 将每个 input batch 分为 R 个 microbatch，attention GPU 和 expert GPU 同时处理不同 microbatch，实现跨 GPU 的 compute-compute 重叠。同时，每 GPU 内维护 2 个通信 stream（dispatch + combine all-to-all）和 1 个计算 stream，通过 CUDA event 同步，实现 compute-communication 重叠。关键洞察：计算和通信重叠是利用了 dispatch 和 combine all-to-all 方向相反、不发生带宽竞争的特性。
    2. **Asymmetric Expert Assignment (Asym-EA)**：当 expert GPU 计算慢于 attention GPU 时（常见于短序列），将部分 expert 计算迁回 attention GPU。基于 "gather and squeeze" 算法（Algorithm 1）：accumulate 跨多层的气泡（bubble = T_E^Exp - T_A^Attn）直到足够 offload 至少一个 chunk 的 experts，然后 squeeze 气泡。最小 offload chunk 由 n_1 = max(1, N/M) 和 n_2 = n_1 · M/N 定义。考虑 memory 约束：通过 α 和 β 系数 enforce attention GPU 内存上限 n_max 和 expert GPU 内存下限 n_min。
    3. **Profiler**：测量 T_A^Attn（attention + gate 在 attention GPU 上的时间）、T_E^Exp（expert 在 expert GPU 上的时间）、T_E^Attn（expert 在 attention GPU 上的时间）。同时测量内存使用以估计 n_min 和 n_max。只需在每个 setup 上运行一次。
  - 实验比较：(1) HeterMoE vs EP (DeepSpeed MoE with Tutel/Lina optimizations) vs DistEP (naïve attention-expert disaggregation without overlapping) vs EP (Ideal, 各 GPU 型号独立运行后求和)；(2) HeterMoE vs heterogeneity-aware Pipeline Parallelism；(3) Ablation: GPU ratio in ZP group, fully homogeneous comparison, Asym-EA effects。

- 后端平台是什么，配置是什么。
  - **On-premise (O1/O2/O3)**：
    - O1: 6× A40 (48GB) + 6× V100 (16GB)
    - O2: 4× A40 (48GB) + 8× V100 (16GB)
    - O3: 6× A40 (48GB) + 3× V100 (16GB)
    - Network: 100 Gbps Mellanox ConnectX-6 RoCE NICs
  - **AWS (C1/C2)**：
    - C1: 2× L40S (48GB, g6e.4xlarge) + 6× T4 (16GB, g4dn.4xlarge)
    - C2: 2× L40S (48GB) + 8× T4 (16GB)
    - Network: 20 Gbps TCP (实际通信占 70% 训练时间，因此模拟 200 Gbps 通过减少 all-to-all 数据量实现)
  - 对比 homogeneous: 2× A100 (80GB, PCIe Gen4)

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 **PyTorch v2.2** (3K 行 Python) + **DeepSpeed v0.14**。
  - 修改内容：
    - **Zebra Parallelism Engine**：在 ZP group 内 split attention 和 expert 模块。初始化时创建 3 个 CUDA stream（2 个通信 + 1 个计算），为每个 microbatch 分配 receive buffer。创建独立的 NCCL dispatch 和 combine all-to-all group。通过 PyTorch NCCL all-to-all wrapper 传入不等 split size 实现不同 GPU 处理不同数量 tokens。
    - **Gate backward 修复**：gate network 的 top-k confidence scores 形成 "residual" 连接，backward 从 MoE block outputs 分两路传播（一路经 confidence scores 到 gate weights，另一路经 expert outputs 到 attention outputs）。HeterMoE 在每层 attention outputs 处停止第二分支的 backward，等待 expert GPU 梯度后再 accumulated 传播到前一层。
    - **Profiler**：从 transformer layer 提取单个 expert FFN，以实际 microbatch 对应的 token 数 B 生成 random tensor，分别在 attention GPU 和 expert GPU 上 profile forward+backward 时间。同时 profile memory usage 以估算 n_min 和 n_max。
  - 评估指标：training throughput (tokens/s)，GPU utilization（有效计算时间百分比），95% confidence intervals。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 论文声明 "We will open source HeterMoE"，截至分析时未找到公开代码仓库。
  - **Zebra Parallelism 执行原理全过程**：

```
┌── ZP Group Setup ──────────────────────────────────────────┐
│ M 个 attention GPU (newer, e.g. A40):                      │
│   - 持有复制: attention blocks + MoE gate + embedding      │
│   - 可选: 通过 Asym-EA 持有部分 experts                    │
│ N 个 expert GPU (older, e.g. V100):                        │
│   - 持有: expert FFN 模块（按 expert parallelism 分布）    │
│ R 个 microbatch: input batch 等分                           │
└────────────────────────────────────────────────────────────┘

┌── Stream Architecture (per GPU) ─────────────────────────┐
│ Stream 0 (compute):  attention/expert 计算                 │
│ Stream 1 (comm D):  dispatch all-to-all                    │
│ Stream 2 (comm C):  combine all-to-all                     │
│ Sync: CUDA events between streams                          │
│                                                             │
│ 例: attention GPU 上                                        │
│   Stream 1: enqueue dispatch A2A kernel                     │
│   event_dispatched = record()                               │
│   Stream 0: cudaStreamWaitEvent(event_dispatched)           │
│             → enqueue attention computation                 │
│             → ...                                           │
└────────────────────────────────────────────────────────────┘

┌── Forward Schedule (Theorem 1) ───────────────────────────┐
│ Attention GPU compute stream:                               │
│   (A_{1,1}^F ... A_{1,R}^F) ...                            │
│   (A_{L,1}^F A_{L,1}^B ... A_{L,R}^F A_{L,R}^B) ...       │
│   (A_{1,1}^B ... A_{1,R}^B)                                │
│                                                             │
│ Expert GPU compute stream:                                  │
│   (E_{1,1}^F ... E_{1,R}^F) ...                            │
│   (E_{L-1,1}^F ... E_{L-1,R}^F)                            │
│   (E_{L-1,1}^B ... E_{L-1,R}^B) ...                        │
│   (E_{1,1}^B ... E_{1,R}^B)                                │
│                                                             │
│ 其中 A_{i,j}^F/B: layer i microbatch j 的 attn fwd/bwd     │
│      E_{i,j}^F/B: layer i microbatch j 的 expert fwd/bwd   │
│                                                             │
│ 依赖约束（以 A_{i,j}^F 为例）:                              │
│   t(A_{i,j}^F) ≥ t(C_{i-1,j}^F) + T_C   (数据依赖)        │
│   |t(A_{i,j}^F) - t(A_{i',j'}^F)| ≥ T_A  (stream顺序)     │
└────────────────────────────────────────────────────────────┘

┌── Overlap Pattern (Zebra) ────────────────────────────────┐
│ Time →                                                      │
│ Attn GPU: [A_{1,1}^F][A_{1,2}^F][A_{1,3}^F][A_{2,1}^F]... │
│                                               ↕ overlap    │
│ Exp GPU:  [  E_{1,1}^F  ][  E_{1,2}^F  ][  E_{1,3}^F  ]...│
│                                                             │
│ 每 GPU 内 compute-communication overlap:                     │
│ Attn GPU: [Dispatch A2A][==== A^F ====][Combine A2A][A^F]  │
│ Exp GPU:  [==== E^F ====][Dispatch A2A][==== E^F ====]... │
│                                                             │
│ 关键：Dispatch和Combine走相反方向，在独立stream上不冲突     │
└────────────────────────────────────────────────────────────┘

┌── Asym-EA "Gather and Squeeze" (Algorithm 1) ────────────┐
│ Input: n (experts/layer), L (layers), M, N (GPU counts)    │
│        T_A^Attn, T_E^Attn, T_E^Exp (profiled times)         │
│                                                             │
│ n_1 = max(1, N/M)    // 每个 Attn GPU 至少 acquire 的 experts│
│ n_2 = n_1 · M/N       // 每个 Exp GPU 至少 offload 的 experts│
│ T_gather = T_E^Exp - T_A^Attn   // 每层每 microbatch 的气泡  │
│ T_squeeze = T_E^Exp·N/n·n_1 + T_E^Attn·N/n·n_2            │
│            // offload 一个 chunk 可消除的气泡                │
│                                                             │
│ t_bubble = 0                                                │
│ for l = 1 to L:                                             │
│   t_bubble += α·β·T_gather    // gather 气泡（含memory约束）│
│   if t_bubble ≥ T_squeeze:                                  │
│     chunks = floor(t_bubble / T_squeeze)                    │
│     o_l = chunks · n_2        // 该层 offload 的 expert 数  │
│     t_bubble -= chunks · T_squeeze                          │
│   else:                                                     │
│     o_l = 0                                                 │
│                                                             │
│ Memory约束:                                                 │
│   α = min(floor(n_max/n_2)·T_squeeze / (L·T_gather), 1)    │
│   β = max(ceil(n_min/n_2)·T_squeeze / (L·T_gather), 1)     │
│   (α和β 至多一个被激活，取决于offload量在上下界之间)       │
└────────────────────────────────────────────────────────────┘
```

  - **Profiler 评估原理**：
    1. 从 transformer layer 提取单个 expert FFN
    2. 根据 global batch size、seqlen、microbatch 数、ZP group setup 计算每个 expert GPU 处理的 token 数 B
    3. 生成 batch=B 的 random tensor → profile forward+backward 时间
       - 在 attention GPU 上: 得到 T_E^Attn
       - 在 expert GPU 上: 得到 T_E^Exp
    4. 提取 attention blocks + MoE gate → profile on attention GPU → T_A^Attn
    5. Memory profiling: 在 expert GPU 上构造单 expert FFN + dummy input → 测量 forward+backward 的 activation/weight/gradient/optimizer state 内存 → 估算 n_min（必须 offload 的 expert 数）；在 attention GPU 上构造不含 expert 的模型 → 估算 n_max（最多可持有的 expert 数）
  - **关键性能数据**：
    | Setting | HeterMoE vs EP | vs DistEP | vs EP(Ideal) |
    |---------|---------------|-----------|-------------|
    | O1/O2/O3 avg (4K) | +22% | +79% | +18% |
    | O1/O2/O3 avg (16K) | +67% | +69% | — |
    | O1/O2/O3 avg (32K) | +89% | +69% | — |
    | AWS avg | +189% | +96% | +17% |
    | vs homogeneous: 2A40+2V100 = 95% of 4xA40 throughput on avg |

## FlowMoE: A Scalable Pipeline Scheduling Framework for Distributed Mixture-of-Experts Training

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - FlowMoE 包含三个 kernel 调度/运行时计算层面的实现：
    1. **Unified Pipeline Scheduling（统一流水线调度）**：将 Transformer block 内的全部任务——MHA 计算、gating 路由、expert 计算、all-to-all (A2A) 通信——统一纳入流水线调度。将每个 Transformer block 的输入 tensor 按 **R 个等分** 切分，除 all-reduce 外的所有计算和通信任务均拆分为 R 个独立子任务，通过统一的 feed-forward 和 backward 顺序编排，使 MHA、gating、expert 和 A2A 在同一条流水线上交错执行。
    2. **Tensor Chunk-Based Priority Scheduling（张量分块优先级调度）**：在反向传播期间，将每层 all-reduce 的梯度张量切成大小为 **S_p** 的 chunk，放入通信任务池。A2A 任务具有最高优先级，all-reduce chunk 仅在没有 A2A 任务 pending 时立即执行，填充通信间隙，最大化计算-通信重叠。
    3. **Bayesian Optimization Auto-Tuning（贝叶斯优化自动调参）**：轻量级 BO 自动搜索最优 all-reduce chunk 大小 S_p。仅需约 8 次采样即可收敛到近优值（如 BERT-Large-MoE 上 ~2.5MB），开销 < 1% 迭代时间。硬件环境变化时重新执行。
  - 实验比较：FlowMoE vs vanillaEP、FasterMoE、Tutel、FSMoE、ScheMoE 在 4 个真实 MoE 模型上的 per-iteration training time、energy consumption、memory usage，以及 675 个自定义 MoE 层配置上的加速比。消融实验对比 Pipe-MoE (仅 MoE 层流水线)、Pipe-AT (加入 MHA+gating)、Pipe-AR (加入 all-reduce, w/ w/o BO) 的逐模块贡献。

- 后端平台是什么，配置是什么。
  - Cluster 1: 2 节点 × 8 × NVIDIA RTX 3090 (24GB)，共 16 GPU，100Gb/s 跨节点网络，Intel Xeon Gold 6248R CPU。
  - Cluster 2: 4 节点 × 2 × NVIDIA RTX 2080Ti (11-12GB)，共 8 GPU，10Gb/s 跨节点网络，Intel Xeon Gold 5118 CPU。
  - 所有参数和梯度使用 32-bit 单精度浮点。

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 **PyTorch** 构建，利用 **Tutel** 优化通信（Tutel 是集成到 PyTorch 中的 MoE 加速库，支持通信和计算任务的异步执行，被 DeepSpeed 作为默认 MoE 训练模块）。
  - 修改内容：
    - **新增三个队列**：DataQueue（任务间数据传递）、A2AQueue（all-to-all 通信任务）、ARQueue（all-reduce 通信任务）。
    - **新增通信池管理器**：在后台线程运行，优先级逻辑确保 A2A 任务优先执行，all-reduce chunk 填充间隙。
    - **Tensor 分区**：将 all-reduce 张量按 S_p 切块。
    - **贝叶斯优化集成**：自动调优 S_p。
  - 性能指标测量：per-iteration training time（平均 1000 次迭代）、energy consumption（NVIDIA SMI 每 5ms 采样，时域积分）、memory usage（NVIDIA SMI 每 1s 监控）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - **开源情况**：代码开源在 https://github.com/ZJU-CNLAB/FlowMoE
  - **FlowMoE 统一流水线调度执行原理全过程**：
    ```
    ┌── Kernel/Scheduling Input ──────────────────────────────────────┐
    │ 每个 Transformer block 的输入 tensor: x^(l) [B, N, H]           │
    │ R: 流水线度（pipelining degree），通常 R=2                        │
    │ S_p: all-reduce 切块大小（BO 自动调优）                          │
    │                                                                  │
    │ 前向任务集（每层 l）:                                            │
    │   AT_r^(l): MHA + gating 计算子任务                              │
    │   D_r^(l):  Dispatch A2A 子任务                                  │
    │   E_r^(l):  Expert 计算子任务                                    │
    │   C_r^(l):  Combine A2A 子任务                                   │
    │                                                                  │
    │ 反向任务集（每层 l）:                                            │
    │   AT_r'^(l): MHA + gating 反向子任务                             │
    │   D_r'^(l):  Dispatch A2A 反向子任务                             │
    │   E_r'^(l):  Expert 反向子任务                                   │
    │   C_r'^(l):  Combine A2A 反向子任务                              │
    │   AR^(l):    All-reduce 梯度子任务（切成 chunk）                  │
    └──────────────────────────────────────────────────────────────────┘

    ┌── 前向调度 ───────────────────────────────────────────────────┐
    │ 顺序: AT_1→AT_2→...→AT_R→E_1→E_2→...→E_R→AT_1^(l+1)→...       │
    │       (计算任务按层流动)                                         │
    │                                                                    │
    │        D_1→...→D_R→C_1→...→C_R→D_1^(l+1)→...                     │
    │       (A2A 通信任务按层流动，与计算交错)                          │
    └──────────────────────────────────────────────────────────────────┘

    ┌── 反向调度（核心创新）─────────────────────────────────────────┐
    │ 计算任务:                                                         │
    │   E_R^(l+1)→...→AT_1^(l+1)→E_R^(l)→...→E_1^(l)→AT_R^(l)→...→AT_1^(l)
    │                                                                    │
    │ A2A 任务:                                                         │
    │   C_R^(l+1)→...→D_1^(l+1)→C_R^(l)→...→C_1^(l)→D_R^(l)→...→D_1^(l)
    │                                                                    │
    │ All-Reduce Chunk 插入（Theorem 1）:                              │
    │   在 A2A 任务的间隙中插入 all-reduce chunk:                       │
    │   C_j^(l) → [AR_chunk if A2A idle] → D_j^(l) → ...               │
    │   优先级: A2A tasks > all-reduce chunks                           │
    │   更小的 S_p → 更细粒度的 gap filling → 更优（Theorem 2）        │
    └──────────────────────────────────────────────────────────────────┘

    ┌── Baseline vs FlowMoE 调度对比 ───────────────────────────────┐
    │                                                                   │
    │ Tutel/ScheMoE (仅 MoE 层流水线):                                 │
    │ Time →                                                            │
    │ [MHA][Gate][==== A2A Dispatch + Expert + A2A Combine =====]       │
    │                                         [All-Reduce]              │
    │  // MHA 和 All-Reduce 串行，占 30-40% 迭代时间                   │
    │                                                                   │
    │ FlowMoE (全 block 流水线):                                        │
    │ Time →                                                            │
    │ [AT_1][AT_2][E_1][E_2][AT_1^(l+1)][E_1^(l+1)]...          (前向) │
    │ [D_1][D_2][C_1][C_2][D_1^(l+1)][C_1^(l+1)]...             (A2A)  │
    │         [AR_chunk_1]      [AR_chunk_2]    [AR_chunk_3]    (AR)   │
    │  // MHA、A2A、AR 全重叠，消除 30-40% 串行开销                    │
    └──────────────────────────────────────────────────────────────────┘
    ```

  - **Bayesian Optimization 调参原理**：
    1. **目标函数**：f(S_p) = per-iteration training time（平均 10 次迭代）
    2. **采样**：随机初始化若干 (S_p, time) 对
    3. **GP 模型**：高斯过程拟合 f(S_p) 的 posterior distribution
    4. **采集函数**：Expected Improvement (EI) 选择下一个采样点
    5. **终止**：约 8 次采样后收敛，BO 开销 < 1% 迭代时间
    6. **输出**：最优 S_p（如 BERT-Large-MoE 上 ~2.5MB）

  - **评估数据集与模型**：
    - GPT2-Tiny-MoE (M=256, H=512, L=12, E=P, k=2, OpenWebText)
    - BERT-Large-MoE (M=512, H=1024, L=24, E/P=2, k=1, wikitext-103)
    - LLaMA2-MoE (M=1024, H=4096, L=32, E=P, k=1, wikitext-103)
    - LLaMA2-MoE-L (M=1024, H=4096, L=64, E=P, k=1, wikitext-103)
    - DeepSeek-V2-S (M=5120, H=1536, L=4, E/P=2, k=8, OpenWebText)
    - DeepSeek-V2-M (M=5120, H=1536, L=7, E/P=2, k=1, OpenWebText)
    - 675 个自定义 MoE 层配置：B∈{2,4,8}, f∈{1.0,1.1,1.2}, N∈{512,1024,2048}, M∈{512..8192}, H∈{512..8192}

  - **关键性能数据**（Cluster 1, 16 × RTX 3090）：
    | Model | vanillaEP | FasterMoE | Tutel | FSMoE | ScheMoE | FlowMoE |
    |-------|-----------|-----------|-------|-------|--------|---------|
    | GPT2-Tiny-MoE | 169.5ms | 135.3ms | 129.3ms | 114.8ms | 116.4ms | **95.6ms** |
    | BERT-Large-MoE | 537.8ms | 490.8ms | 501.1ms | 421.9ms | 405.6ms | **351.9ms** |
    | LLaMA2-MoE | 1987.7ms | 1759.1ms | 1534.1ms | 1292.6ms | 1374.3ms | **1124.0ms** |
    | DeepSeek-V2-S | 5843.3ms | 4562.5ms | 4481.4ms | 3895.6ms | 4093.7ms | **3205.3ms** |
    
  - **消融实验**（M=8192, H=8192, 16 GPU）：
    | Configuration | Pipe-MoE | Pipe-AT | Pipe-AR | Time | Speedup vs vanillaEP |
    |--------------|----------|---------|---------|------|---------------------|
    | vanillaEP | ✗ | ✗ | ✗ | 1630.8ms | 1.00× |
    | Tutel | ✓ | ✗ | ✗ | 1115.2ms | 1.46× |
    | FlowMoE-AT | ✓ | ✓ | ✗ | 1012.6ms | 1.61× |
    | FlowMoE-AR | ✓ | ✗ | ✓ (w/o BO) | 971.5ms | 1.68× |
    | FlowMoE-AR(BO) | ✓ | ✗ | ✓ (w/ BO) | 895.3ms | 1.82× |
    | FlowMoE | ✓ | ✓ | ✓ | **796.1ms** | **2.05×** |

## FlashMoE: Fast Distributed MoE in a Single Kernel

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - FlashMoE 在 kernel 调度/运行时计算层面的核心实现是将整个分布式 MoE 操作——Gate、Dispatch、Expert FFN、Combine 以及跨 GPU 通信——融合为**单个持久 GPU kernel**（megakernel），仅需一次 kernel launch：
    1. **Actor-based 持久 Kernel 架构**：将 GPU thread block 和 warp 特化为三种角色——Processor（N-1 个 block，执行 GEMM 和 element-wise 计算）、Scheduler（1 个 warp，多线程、work-conserving 的 tile 级任务调度器）、Subscriber（3 个 warp，解码来自 peer GPU 的 tile packet 为 task descriptor）。OS block（包含 Scheduler + Subscriber）占用最少资源用于管理任务，其余 block 全力执行 MoE 计算。
    2. **Tile-Level Parallelism and In-Kernel Scheduling**：输入 token 矩阵被分割为独立的 tiles（128×64），每个 tile 映射到一个 task descriptor。Scheduler 基于 readiness 动态将 task 分配给 Processor block，确保无 SM 空闲。Subscriber 解码远程 tile packet → 通知 Scheduler → Scheduler 分发 task → Processor 执行 GEMM0/GEMM1/Combine。
    3. **Device-Initiated One-Sided Communication via NVSHMEM**：摒弃传统的同步 AlltoAll collective，改用 NVSHMEM 建立跨 GPU 的全局地址空间（PGAS 模型），实现 GPU 直接发起的 (R)DMA 传输。设计 Symmetric Tensor Layout（维度 P×R×B×E×C×H）通过 temporal buffering 确保所有写入无冲突（Theorem 3.1 证明），实现完全非阻塞的一对一通信。
    4. **Payload-Efficient Communication via In-Place Padding**：在本地 symmetric tensor buffer 中进行 in-place padding（将 expert capacity 对齐到 tile block size bM=128），消除传统实现中为满足 collective 对称性而进行的零填充 token 网络传输，节省通信带宽和避免无效计算。
    5. **Custom GEMM via CUTLASS**：在 kernel 内集成 CUTLASS 实现高性能 GEMM 操作，包括 fused GEMM + epilogue + async tile staging/transfer。
  - 实验比较：FlashMoE (FP32) vs Comet (FP16, cudaMemcpyPeerAsync)、FasterMoE (FP16, NCCL)、Megatron-CUTLASS (FP16, NCCL)、Megatron-TE (FP16, Transformer Engine with NCCL)。DeepEP+Megatron-LM 出现在 GPU utilization 对比中。评估指标：Forward Latency、GPU SM Utilization、Throughput、Overlap Efficiency（弱扩展效率）、Expert Scalability。

- 后端平台是什么，配置是什么。
  - GPU 后端：8× NVIDIA H100 80GB GPU，NVLink 互联（单节点内）
  - CPU：20 vCPUs，125 GB RAM
  - 软件环境：PyTorch 2.6.0，CUDA 12.8，Ubuntu 22.04
  - 通信库：NVSHMEM v3.2.5（GPU-initiated one-sided (R)DMA）
  - BLAS 后端：CUTLASS（in-kernel 高性能 GEMM）
  - GPU 利用率 profiling：Nsight Systems
  - 额外测试（Table 1, 图 4）：2× A100 GPU, 300 GB/s unidirectional bandwidth

- 评估性能的软件/脚本是什么。修改了什么。
  - 实现为 PyTorch 自定义 C++/CUDA 扩展（6820 行 CUDA/C++），通过 `torch.autograd.Function` 集成。
  - 修改内容：
    - **完全融合的 MoE Megakernel**：将 Gate、Dispatch、Expert FFN (2×GEMM)、Combine、跨 GPU 通信全部融合为单个持久 kernel。仅需 1 次 kernel launch，而 DeepSpeed-MoE 需要 550 次（Table 1）。
    - **Gate Kernel 融合**：`FusedGate` 函数在 kernel 内计算 routing table T_φ 和 gate affinity scores G_φ。
    - **Processor Actor Kernel**（Algorithm 2）：warp 0 从 Scheduler 接收 task → `__syncthreads()` → switch task type → GEMM0（fused GEMM + epilogue + async tile staging）+ GEMM1（fused GEMM + epilogue + async tile transfer）+ Combine。
    - **Scheduler Actor Kernel**（Algorithm 3）：sweep processor/subscriber doorbells → 汇总本地观察到的 task count → warp inclusive sum → 从 ready queue 中发信号给 processor。
    - **Subscriber Actor Kernel**（Algorithm 4）：3 个 warp 并发访问 dispatch flags → atomically 检索信号 → 解码 packet 为 GEMM0 task descriptors（利用 expert weights X）→ 写入 task queue → 通知 Scheduler。类似流程处理 combine signals（利用 T_φ, G_φ, O）。
    - **Symmetric Tensor Layout + NVSHMEM Communication**：建立 L ∈ R^{P×R×B×E×C×H} 的对称张量布局，NVSHMEM put/get 实现 one-sided (R)DMA 传输。
    - **In-Place Padding**：在本地 buffer 中将 expert capacity 对齐到 tile block size bM=128，消除网络传输中的零填充。
  - 性能评估脚本：执行 32 warmup passes + 32 timed passes 的 forward pass，取平均 latency。使用 Nsight Systems 测量 SM utilization（SM active cycles / total cycles）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - **开源情况**：代码开源在 https://github.com/osayamenja/FlashMoE
  - **FlashMoE 单一持久 Kernel 执行原理全过程**：

```
┌── Kernel Input ──────────────────────────────────────────────┐
│ A ∈ R^{S×H}: input token matrix                              │
│ X ∈ R^{E×H×D}: expert weights (3-D tensor)                   │
│ N: total number of thread blocks on GPU                      │
│ S: sequence length, H: embedding dim, E: local experts       │
│ D: FFN intermediate dim                                       │
└──────────────────────────────────────────────────────────────┘

┌── Single Kernel Launch (Algorithm 1) ───────────────────────┐
│ // 仅一次 kernel launch，替代 baseline 的 33-550 次 launch   │
│                                                              │
│ // Step 1: Fused Gate (所有 block 参与)                       │
│ T_φ, G_φ ← FusedGate(A)                                      │
│ // T_φ ∈ (ℕ×ℝ)^{E×C}: routing table                          │
│ //   T_φ(e,c) = (token_idx, combine_weight)                  │
│ // G_φ ∈ R^{S×E}: gate affinity scores                       │
│                                                              │
│ // Step 2: Actor role assignment                             │
│ if blockId + 1 < N:  // N-1 Processor blocks                 │
│     Dispatch(T_φ, A)  // prepare dispatch packets            │
│     processor::start()                                       │
│ else:  // 1 OS block (4 warps)                               │
│     if warpID == 0: scheduler::start()                       │
│     else (warps 1-3): subscriber::start(T_φ, G_φ, O, X)     │
└──────────────────────────────────────────────────────────────┘

┌── Actor Interaction Chain (Figure 6) ───────────────────────┐
│                                                              │
│  D_j^i ──Dispatch──▶ S_b^i ──Tasks──▶ S_h^i ──Schedule──▶   │
│                    (Subscriber)   (Scheduler)                │
│                                                              │
│  ──GEMM₀──▶ P^i ──Notify──▶ S_h^i ──GEMM₁──▶ P^i            │
│          (Processor)   (Scheduler)        (Processor)        │
│                                                              │
│  ──Send──▶ S_b^j ──Notify──▶ S_h^i ──Combine──▶ P^j         │
│        (Remote GPU          (Scheduler)    (Processor)       │
│         Subscriber)                                          │
│                                                              │
│  其中上标 i, j 标识 GPU，D_j^i 表示 GPU j 向 GPU i dispatch  │
│  通知通过 shared memory (Subscriber↔Scheduler) 或            │
│  global memory (Scheduler↔Processor) 传递                    │
└──────────────────────────────────────────────────────────────┘

┌── Processor Actor Execution (Algorithm 2) ──────────────────┐
│ while interrupt == False:                                    │
│     if warpId == 0:  // leader warp                          │
│         awaitTaskFromScheduler(interrupt, signal)            │
│         FencedNotifyRQ(ready)                                 │
│     __syncthreads()                                          │
│     warpReadTQ(tQ, signal, task)                             │
│     __syncthreads()                                          │
│                                                              │
│     switch task.Type:                                        │
│         case GEMM0:                                          │
│             // Fused GEMM, epilogue, async tile staging      │
│             fGET(GEMM0, task)                                │
│             // thread 0 notifies tile completion             │
│             NotifySchedulerNextGEMM(tQ)                      │
│                                                              │
│         case GEMM1:                                          │
│             // Fused GEMM, epilogue, async tile transfer     │
│             fGET(GEMM1, task)                                │
│             // Tile result sent to remote GPU via NVSHMEM    │
│                                                              │
│         case Combine:                                        │
│             // Weighted expert combine                       │
│             combine(task)                                    │
└──────────────────────────────────────────────────────────────┘

┌── Subscriber Actor Execution (Algorithm 4) ─────────────────┐
│ // 3 warps 并发执行                                          │
│ while AtomicLoad(interrupt) == False:                        │
│                                                              │
│     // Phase 1: Dispatch flags                               │
│     do in parallel:                                          │
│         Visit dispatch flags                                 │
│         Atomically retrieve signal                           │
│         if Signal is set and flag not visited:               │
│             Mark visited                                     │
│             SelfCorrectTaskBound(taskBound, Signal)          │
│             Memory fence before consuming packet             │
│             Decode packet → GEMM0 task descriptors           │
│             Write descriptors to task queue (tQ)             │
│             Notify Scheduler of decoded tasks                │
│                                                              │
│     // Phase 2: Combine signals                              │
│     do in parallel:                                          │
│         Visit combine flags (one per tile)                   │
│         if Signal is set and flag not visited:               │
│             Mark visited                                     │
│             Memory fence before consuming packet             │
│             Decode → combine task descriptors                │
│             Write to tQ                                      │
│             Notify Scheduler                                 │
└──────────────────────────────────────────────────────────────┘

┌── Scheduler Actor Execution (Algorithm 3) ──────────────────┐
│ // 1 warp 执行                                               │
│ scheduled = 0                                                │
│ tTB = AtomicLoad(taskBound)  // total task bound             │
│ rQ = PopulateRQ()  // 初始化 ready queue (processor IDs)     │
│                                                              │
│ while scheduled < tTB:                                       │
│     lt = 0                                                   │
│     do in parallel:                                          │
│         Sweep doorbells (processor + subscriber)             │
│         Populate observed task counts → tqState              │
│         Aggregate local task counts → lt                     │
│                                                              │
│     qS, taskTally = 0                                        │
│     WarpInclusiveSum(lt, qS, taskTally)                      │
│                                                              │
│     while taskTally > 0:                                     │
│         Repopulate rQ with ready processor ids               │
│         do in parallel:                                      │
│             Start at rQ[qS], signal processors               │
│             about task indices from tqState                  │
│                                                              │
│     // Thread 0 更新 task bound (动态调整)                    │
│     tTB = WarpBroadcast(AtomicLoad(taskBound))               │
│                                                              │
│ InterruptSubscribers()                                       │
│ InterruptProcessors()                                        │
└──────────────────────────────────────────────────────────────┘

┌── Symm. Tensor Layout + NVSHMEM Communication ──────────────┐
│ L ∈ R^{P×R×B×E×C×H}                                        │
│ P: expert parallel world size                                │
│ R: communication rounds (2: dispatch + combine)              │
│ B: staging buffers (2: outgoing + incoming)                  │
│ E: local expert count                                        │
│ C: upscaled expert capacity                                  │
│ H: token embedding dimension                                 │
│                                                              │
│ Size(L) ≈ 4 × Size(T) (uniform expert distribution)         │
│ 内存开销 ≤ 2% (inference: 0.11%~2.15%, Table 3)              │
│                                                              │
│ 索引保证 (Theorem 3.1):                                      │
│ - 所有写入 index coordinate i = (p*, r, b, e, c)            │
│ - inter-device: p* = p_s, b = 1 (self-loop 也适用)          │
│ - intra-device staging: b = 0 → p_s = p_t                   │
│ - 不同 source process 的写入总是不同的 p* → 无冲突           │
│ - 同一 source 的写入有不同的 c_j → 无冲突                    │
└──────────────────────────────────────────────────────────────┘
```

  - **Task 抽象 (式 4)**：
    ```
    F_t(A, B, C, D) := C ← φ(A ⋆_t B + D)
    t = (M, ⋆, φ)
    // M: metadata (device ID, tile index, etc.)
    // ⋆: binary tensor op (. or ⊙)
    // φ: element-wise activation (ReLU or identity)
    
    // FFN 表达:
    t₁ = (M, ·, φ₁),  t₂ = (M, ·, φ₂)  // φ₂ = identity
    // Combine 表达:
    t₃ = (M, ⊙, φ₂)
    ```

  - **评估原理**：
    1. **Forward Latency (图 8)**：配置 MoE transformer（16 attention heads, H=2048, D=2048），top-2 routing, capacity factor=1.0。DDP + Expert Parallelism。在 4 GPU 和 8 GPU 上测试不同 sequence lengths (up to 16K tokens)，32 warmup + 32 timed 取平均值。
    2. **GPU Utilization (图 9)**：Nsight Systems 测量 SM utilization = active cycles / total cycles × 100%。100 次 iteration 取平均值。FlashMoE 达 93.17%，DeepEP+Megatron-LM 仅 13.55%。
    3. **Throughput (图 10)**：计算为 (T × N_G) / latency，T=tokens per GPU，N_G=GPU count。测量 2/4/8 GPU 扩展性。
    4. **Overlap Efficiency (图 11)**：弱扩展效率 O_e = T(2) / T(N_G)，T(N_G) 为 N_G GPU 时的延迟。FlashMoE 在 8 GPU 时仍维持高效率，baseline 在 ≥4 GPU 时降至 50% 以下。
    5. **Expert Scalability (图 12)**：固定 T=16K tokens，expert 数从 8 扩展到 128（总 expert 数，每 GPU 1/8）。FlashMoE 保持平坦低延迟。

  - **关键性能数据**：
    | Metric | FlashMoE (FP32) | Best Baseline (FP16) | Improvement |
    |--------|-----------------|---------------------|-------------|
    | Forward Latency (8 GPU, 16K) | — | Megatron-TE | **6.4×** speedup |
    | GPU SM Utilization | **93.17%** | Megatron-TE (59.11%) | **9×** vs FasterMoE |
    | Throughput (8 GPU) | 17.7 MT/s | FasterMoE | **5.7×** |
    | Overlap Efficiency (8 GPU) | — | Megatron-CUTLASS | **4×** |
    | Expert Scalability (128 experts) | — | — | **6.6×** (8 H100) |

  - **Implementation Metrics (Table 4)**：
    - Total LOC: 6820 (CUDA/C++)
    - Kernel stack frame: 0 B
    - Spill stores/loads per thread: 0
    - Shared memory per block: 46 KB
    - Registers per thread: 255
    - Max active blocks per SM: 2
    - Compilation time: 53s
    - Binary size: 29 MB

  - **Kernel Launch 对比 (Table 1, 2 A100 GPUs, 32 experts/GPU)**：
    | Implementation | GPU Ops per MoE Layer |
    |---------------|----------------------|
    | FlashMoE | **1** |
    | COMET | 33 |
    | Megatron-LM CUTLASS | 85 |
    | Megatron-LM TE | 261 |
    | Megatron-LM + DeepEP | 432 |
    | DeepSpeed-MoE | 550 |

## HD-MoE: Hybrid and Dynamic Parallelism for Mixture-of-Expert LLMs with 3D Near-Memory Processing

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - HD-MoE 在 kernel 调度/运行时计算层面的实现是 **Dynamic Placement Strategy（在线动态专家调度）**，包含三个关键组件：
    1. **Priority Detection and Computation Prediction**：利用相邻层专家激活的时间局部性预测下一层计算热点。对每个节点 c 上的专家 i，计算优先级分数 `prio_ic = 2 * P_ic * f̂_i * IS / comp`，其中 f̂_i 是预测的激活频率。选择最高负载节点上优先级最高的专家作为预广播候选。
    2. **Optimal Broadcast Chunk Size**：基于 α-β 通信模型推导最优广播 chunk size `c = sqrt(α * h * IS / (2 * β * k * sqrt(D)))`，在给定 runtime window（由上一层推理延迟决定）内最大化预广播效率。
    3. **Communication-Efficient Dispatch**：预广播后，每个 token 被路由到持有其激活专家的任意节点中当前计算负载最低的节点，在不引入额外通信开销的前提下最小化负载不均衡。
  - 实验比较：静态部署策略 vs 动态调度策略在不同推理场景（math/coding/reasoning 等 MT Bench 问题类型）下的延迟和加速比。两种配置：(5 TFLOPS, 50 GB/s, batch=512, 预广播 2 experts/layer) 和 (2.5 TFLOPS, 75 GB/s, batch=512, 预广播 5 experts/layer)。广播 2 experts 时动态策略平均加速 1.15×，广播 5 experts 时平均加速 1.25×。

- 后端平台是什么，配置是什么。
  - 模拟的 3D NMP 加速器，具有可配置的计算吞吐和通信带宽：2.5 TFLOPS / 75 GB/s，5 TFLOPS / 50 GB/s，10 TFLOPS / 25 GB/s。
  - 2D mesh NoC 拓扑：4×4, 4×8, 8×8 节点网格。
  - 模型：Mixtral-8x7B-Instruct, DeepSeek-V2-Lite-Chat, Qwen2-57B-A14B-Instruct。

- 评估性能的软件/脚本是什么。修改了什么。
  - 论文自建 Python 离散事件模拟器，模拟 2D mesh NoC 中不规则 all-to-all 通信。实现 XY routing + 优先级队列事件调度 + 链路占用追踪。
  - 线性规划（LP）求解器用于 Node Balance 优化；Bayesian Optimization 用于 Link Balance 物理映射。
  - 验证工具：ASTRA-sim [27] 用于验证 ring all-reduce 延迟模型的准确性。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 代码开源：https://github.com/angerybob/HD-MoE
  - 动态调度评估原理：
    1. **输入**：专家激活 trace（从 MT Bench 数据集不同问题类型采样），placement matrix P_ic（offline 确定），预测的激活频率 f̂_i。
    2. **Runtime 调度流程**：每层推理开始时 → Priority Detection 扫描各节点负载，识别计算热点 → 选择最高负载节点上优先级最高的 expert → 按最优 chunk size 将其预广播到所有节点 → 每个 token 根据其激活的 experts 从候选节点中选择当前负载最低的节点 dispatch → 节点执行本地 expert 计算 → 进入下一层。
    3. **性能测量**：模拟器记录每层的 computation latency（max across nodes）和 communication latency（discrete-event 调度时间线），计算 MoE Decomposed Latency = t_comp + t_comm。Normalized TBT = 当前策略 TBT / TP baseline TBT。
    4. **动态 vs 静态对比**：静态策略使用 reasoning 问题确定固定 placement；动态策略在运行时根据不同问题类型（math/coding/writing 等）自适应调整广播和调度决策。记录各场景下 per-MoE-layer latency 和 speedup。

## Lazarus: Resilient and Elastic Training of Mixture-of-Experts Models with Adaptive Expert Placement

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - Lazarus 在 kernel 调度/运行时计算层面的核心实现：
    1. **Flexible Token Dispatcher CUDA Kernel（Algorithm 1）**：将整个 MoE 层的 token dispatch 实现为单个 CUDA kernel，对所有 experts 和 target ranks 并行处理。核心逻辑：(a) 根据 expert 的 replica 数 r_e 和每 replica 应处理的 token 数 p_e = t_e/r_e，计算每个 rank 对每个 expert 的处理容量 P_{e,j} = p_e × R_{e,j}；(b) 优先将 rank j 本地已有的 token 分配给它自身（min(P_{e,j}, T_{e,j})）；(c) 将超出本地容量的剩余 token 按各 rank 的剩余容量比例分配到其他 rank（(T_{e,i} - D_{e,i}) × P_{e,j} / Σ_k P_{e,k}）；(d) 根据 dispatch schedule 将输入 activation h 重排（reshuffle）为连续 buffer h'，使 routed to same expert + dispatched to same rank 的 token 连续排列，供后续 all-to-all collective 使用。
    2. **Adaptive Expert Replica Allocation（Eq. 1）**：运行时根据 expert load 分布动态计算每个 expert 应分配的 replica 数 r_e = max{⌊t_e / Σ_{e'=e}^{E} t_{e'} × (N·c - Σ_{e'=1}^{e-1} r_{e'})⌋, f}，其中 t_e 为 routed token 数，N 为节点数，c 为每节点 replica 槽位数，f 为容错阈值。同时保证 Σ_e r_e = N·c, r_e ≥ f（支持 <f 个节点故障时 100% 恢复）。
    3. **Maximum Rank Overlap (MRO) 专家放置算法**：将 experts 按流行度分为 ⌈E/c⌉ 组，每组内最大化 experts 跨节点的重叠度（S_{c*(i-1)+1} ⊂ S_{c*(i-1)+2} ⊂ ... ⊂ S_{c*i}），使某组 representative expert 的 replica 存在即可恢复全组。定理证明 MRO 在均匀随机节点故障下最大化恢复概率。
    4. **Efficient Reconfiguration Runtime**：故障后利用贪心算法最小化迁移的 replica 数，将物理节点映射到新 placement plan 中重叠度最大的节点，跨节点并行获取缺失 expert states。
  - 实验比较：Lazarus vs DeepSpeed MoE (DS)、DS(FT)（使用 Lazarus runtime 的容错版本），以及 Tutel、Tutel(FT) 在 controlled failures（单节点/多节点）、spot instance traces 下的吞吐量和总训练样本数。消融实验：单 MoE layer 在不同 expert load ratio 下的吞吐量对比 + 恢复概率对比。

- 后端平台是什么，配置是什么。
  - **本地集群**：5 台服务器，每台 2× NVIDIA RTX 3090 GPU + 100 Gbps Mellanox ConnectX-5 NIC，100 Gbps Mellanox SN2100 switch。每 GPU 视作独立节点模拟 10 GPU 集群。NFS server 通过 10 Gbps NIC 连接用于 checkpoint 存储。
  - **AWS 集群**：16× g5.2xlarge instances，每实例 1× NVIDIA A10G GPU，10 Gbps TCP 网络。AWS EFS 共享文件系统存储 checkpoint。使用 gradient accumulation (step=20) 减少频繁梯度同步。
  - **模拟环境**：模拟 DeepSeek V3 模型训练，每节点 8× H200 GPU + 8× 400 Gbps NIC。使用 DeepSeek-V3/R1 Performance Simulator (github.com/zartbot/shallowsim) 的性能模型。
  - 软件：PyTorch v2.3，DeepSpeed v0.13（组件复用），CUDA，NCCL v2.12.12。

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 **PyTorch** (v2.3) 实现，使用 **DeepSpeed** (v0.13) 组件。总计 4K LoC Python + 500 LoC CUDA。
  - **修改/新增内容**：
    1. **Flexible Token Dispatcher CUDA Kernel (~500 LoC CUDA)**：实现 Algorithm 1 的 CUDA kernel，在 MoE block forward path 中替代原有 DeepSpeed MoE 的 dispatch 逻辑。计算 dispatch schedule → reshuffle input activations → 执行 flexible all-to-all（无 padding）。
    2. **Controller + Agent 架构（Python async）**：Controller（CPU-only 节点）管理集群全局状态，通过 TCP socket 与各 GPU 节点的 Agent 通信。Controller 执行 MRO placement 算法（<100ms 计算时间），Agent 周期性收集 expert routing history 并 relay placement plan 给 worker。
    3. **Lazarus Runtime**：基于 placement plan 配置 NCCL communication groups（expert/non-expert gradients all-reduce + all-to-all）。Data Parallelism + Expert Parallelism with adaptive placement。Batched NCCL send/recv 用于 reconfiguration 时的 state transfer。
    4. **Routing History Trace Replay**：使用 SmartMoE artifact 的 routing history trace 模拟 gate network routing decision，保证可复现性。
  - 评估指标：training throughput (samples/sec)，total trained steps/samples，reconfiguration time，state transfer size/time，recovery probability（理论枚举所有故障组合）。
  - 模拟评估：自建 simulator，使用 constant node preemption probability + variable new node allocation probability per simulation hour。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - **开源情况**：论文声明 "We will open source Lazarus"，截至分析时（2026/05）未找到公开代码仓库。
  - **Flexible Token Dispatcher CUDA Kernel 执行原理全过程**：
```
┌── Kernel Input ──────────────────────────────────────────┐
│ N: number of GPUs; i: current GPU rank                    │
│ R_{e,j}: number of replicas for expert e assigned to     │
│         rank j (from placement plan)                      │
│ T_{e,j}: number of tokens routed to expert e at rank j   │
│         (collected via all-gather of E integers/rank)     │
│ h: activation of input tokens to MoE block                │
│ E: number of experts                                      │
└──────────────────────────────────────────────────────────┘

┌── Step 1: Compute per-expert per-replica token quota ────┐
│ Parallel for e ← 0 to E:                                  │
│     r_e = Σ_j R_{e,j}  // total replicas for expert e    │
│     t_e = Σ_j T_{e,j}  // total tokens routed to e       │
│     p_e = t_e / r_e    // tokens each replica handles    │
│                                                           │
│ Parallel for j ← 0 to N:                                  │
│     P_{e,j} = p_e × R_{e,j}  // rank j capacity for e   │
│     P_{e,j} = P_{e,j} - min(P_{e,j}, T_{e,j})           │
│     // subtract tokens already local to j                │
│     // remaining P_{e,j} = residual capacity             │
│                                                           │
│ D_{e,i} = p_e × R_{e,i} - P_{e,i}  // locally processed │
└──────────────────────────────────────────────────────────┘

┌── Step 2: Distribute overflow tokens ────────────────────┐
│ Parallel for j ← 0 to N, j ≠ i:                           │
│     D_{e,j} = (T_{e,i} - D_{e,i}) ×                      │
│               P_{e,j} / Σ_{k≠j} P_{e,k}                 │
│     // distribute rank i's remaining tokens              │
│     // proportionally to other ranks' residual capacity  │
└──────────────────────────────────────────────────────────┘

┌── Step 3: Compute dispatch counts per rank ──────────────┐
│ Parallel for j ← 0 to N:                                  │
│     s_j = Σ_e D_{e,j}  // total tokens to rank j        │
└──────────────────────────────────────────────────────────┘

┌── Step 4: Reshuffle activations ─────────────────────────┐
│ Parallel for j ← 0 to N:                                  │
│     Parallel for e ← 0 to E:                              │
│         start = Σ_{0..j-1} s_{j'} + Σ_{0..e-1} D_{e',j}│
│         end = Σ_{0..j-1} s_{j'} + Σ_{0..e} D_{e',j}    │
│         h'[start..end] = tokens in h routed to e that   │
│            are dispatched to rank j, starting from the   │
│            (Σ_{j'=0}^{j-1} D_{e,j'})-th token           │
│                                                           │
│ // h' is now sorted by (target_rank, expert_id)          │
└──────────────────────────────────────────────────────────┘

┌── Step 5: Flexible All-to-All ───────────────────────────┐
│ // Unlike vanilla EP's padded all-to-all                  │
│ // each rank j receives exactly s_j tokens                │
│ // (no padding, no wasted communication)                  │
│ Dispatch all-to-all: h' → remote ranks (s_j tokens each) │
│ Expert computation on received tokens                     │
│ Combine all-to-all: results → original ranks              │
└──────────────────────────────────────────────────────────┘
```

  - **MRO Expert Placement 执行原理全过程**：
```
┌── Input ────────────────────────────────────────────────┐
│ E: experts, sorted by popularity (ascending)             │
│ r_e: replica count for each expert e                     │
│ N: number of nodes                                       │
│ c: replica slots per node                                │
└──────────────────────────────────────────────────────────┘

┌── Case 1: E ≤ c (simple case) ─────────────────────────┐
│ Strategy:                                                │
│ - First r_1 nodes: place experts {1, 2, ..., E}         │
│ - First r_2 nodes: place experts {2, ..., E}            │
│ - ...                                                    │
│ - First r_E nodes: place expert {E}                     │
│ Result: S_1 ⊂ S_2 ⊂ ... ⊂ S_E                          │
│ Recovery probability = P(any of first r_1 nodes alive) │
│ This is the theoretical upper bound → optimal            │
└──────────────────────────────────────────────────────────┘

┌── Case 2: E > c (difficult case) ──────────────────────┐
│ Step 1: Partition experts into ⌈E/c⌉ groups:            │
│         Group 1: experts {1, ..., c}                     │
│         Group 2: experts {c+1, ..., 2c}                  │
│         ...                                              │
│                                                          │
│ Step 2: Partition first ~nodes:                          │
│         Group 1 gets r_1 nodes                           │
│         Group 2 gets r_{c+1} nodes                       │
│         ...                                              │
│                                                          │
│ Step 3: For each group i, place experts                  │
│         in group i on its assigned nodes                 │
│         using the simple case strategy                   │
│         → S_{lo} ⊂ ... ⊂ S_{hi} within group           │
│                                                          │
│ Step 4: Fill vacant slots with remaining replicas       │
│                                                          │
│ Result: Recovery requires ≥1 node alive in each         │
│         group's representative expert set               │
│ Optimality: Theorem 1 proves MRO maximizes              │
│         Pr(∪_{a∈A} Col_a = [E]) for given r_e          │
└──────────────────────────────────────────────────────────┘
```

  - **Reconfiguration 流程**：
```
Failure detected by Controller (heartbeat timeout)
  ↓
Controller re-computes expert allocation + MRO placement
  using only remaining alive nodes (<100ms)
  ↓
Greedy node mapping: min(#new experts per node)
  physical_node → placement_plan_node
  ↓
Agent relays new plan → Lazarus runtime
  ↓
NCCL enqueued ops timeout (10~20s)
  ↓
NCCL communication groups reconfigured (5~15s)
  ↓
State transfer: batched NCCL send/recv from owning nodes
  (e.g. GPT-L: 160 expert states, 7.6s transfer time)
  ↓
Training resumes with all remaining GPUs fully utilized
Total reconfiguration time: 20~40s
```

  - **关键性能数据**：
    | 场景 | GPT-S (521M, 8E) | GPT-L (1.7B, 16E) |
    |------|-------------------|--------------------|
    | 5min MTBF vs DS | 2.8× | 5.7× |
    | 40min MTBF vs DS | 1.6× | 2.3× |
    | 5min MTBF vs DS(FT) | 1.4× | 2.8× |
    | Spot trace vs DS | 2.3× | 3.4× |
    | Spot trace vs DS(FT) | 1.2× | 1.8× |
    | Laz. throughput (no failure, GPT-M) | 45 samples/s | DS: 34 samples/s |
    | 4-node failure recovery prob (GPT-L) | 41% (Lazarus) | 12% (spread) |

## JANUS: Disaggregating Attention and Experts for Scalable MoE Inference

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - JANUS 包含两个 kernel 调度/运行时计算层面的实现：
    1. **Activated-Expert-Balanced Scheduling (AEBS) GPU Kernel（Section 3.4, Algorithm 1）**：实现为 GPU kernel，在每 MoE 层解码时执行。调度流程：Gating 产生 top-K 逻辑 expert IDs → Kernel 扫描路由结果收集激活 expert 集合 (GPU 线程并行处理 tokens) → 单副本 expert 分配到唯一持有实例 → 多副本 expert 贪心选负载最低实例 (load[g] = 当前层已分配给实例 g 的 unique expert 数) → 将每个 token 的路由结果从逻辑 EID 重写为物理 RID → Dispatch token。每个 MoE instance 独立执行相同 kernel（synchronization-free），通过确定性算法 + 相同输入保证一致性，消除跨 GPU 协调开销。
    2. **NVSHMEM-based One-Sided Communication Kernel（Section 3.3, Implementation）**：使用 NVSHMEM putmem_signal/signal_wait 原语实现 GPU-initiated one-sided RDMA——发送端 GPU kernel 直接写入接收端 GPU memory → signal 通知完成。元数据 (layer index, token count) 打包进 signal value 避免单独传输。NVSHMEM 参数调优包括 IBGDA transport、request-batching threshold、per-peer RC queue count。与 NCCL intra-node collectives 配合实现 adaptive two-phase communication。
  - 实验比较：
    - AEBS vs EPLB（DeepSeek EP Load Balancer）：a_max 对比 (Fig. 13)、MoE-layer latency (Fig. 14)、scheduling overhead (Fig. 15)
    - Two-phase (2PC) vs One-phase (1PC) communication：TPOT 和 throughput 消融 (Fig. 12)
    - EGate (MoE 侧 gating) vs AGate (Attention 侧 gating)：throughput 消融 (Fig. 12)
    - AEBS overhead: <20μs (small batch) ~ <90μs (batch=4096)

- 后端平台是什么，配置是什么。
  - GPU：NVIDIA H100 80GB (每节点 8×, 最多 4 节点 32 GPU)
  - Intra-node 互联：NVLink 900 GB/s
  - Inter-node 互联：400 Gbps InfiniBand NIC per GPU + GPUDirect RDMA
  - 通信库：NVSHMEM (GPU-initiated one-sided RDMA) + NCCL (intra-node collectives)
  - 软件：CUDA/C++ 自定义 kernel (~300 行) + Python (~4K 行)

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 SGLang 评估端到端性能（TPOT, TPG）。
  - 微基准脚本测量：
    - AEBS scheduling overhead：测量 AEBS kernel 在不同 batch size (64–4096) 和 MoE 实例数 (8–16) 下的执行时间 (μs)
    - a_max 对比：记录 AEBS vs EPLB 在不同 batch size 和 MoE instance 数下的最大激活 expert 数
    - Communication overhead：消融 1PC vs 2PC vs EGate vs AGate 各组件
    - MoE-layer latency：测量单 MoE 层的 wall-clock 延迟
  - 修改内容：
    - **AEBS GPU Kernel**：CUDA kernel 实现 Algorithm 1，每 MoE 层在每个 MoE instance 的 default stream 上 launch。输入：token routing results (GPU global memory)、replica mapping (GPU constant memory)、instance metadata (更新频率低)。输出：per-token physical replica IDs (GPU global memory)。
    - **NVSHMEM Communication**：替换 SGLang 原有 intra-instance 数据移动为 NVSHMEM putmem_signal/signal_wait。发送端 kernel: prepare payload → nvshmem_putmem_signal(dest_pe, dest_addr, src_addr, size, signal_addr, signal_value)。接收端: nvshmem_signal_wait(signal_addr, expected_value) → 读取 payload。
    - **Metadata Packing**：将 layer index + token count 通过位运算打包为 64-bit signal value，CPU 侧仅在首 MoE 层 unpack 并缓存。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 论文未明确提供 JANUS 独立开源仓库。基于开源 SGLang + NVSHMEM + NCCL 实现。
  - **AEBS GPU Kernel 执行原理全过程**：

    ```
    ┌── Kernel Input (GPU Global Memory) ───────────────────────┐
    │ L(i,j): logical EID for token i, expert j  [T × k, int]   │
    │ R(e): number of replicas for expert e       [E, int]       │
    │ G(e): set of instances hosting replicas     [E, list]      │
    │ P(e,g): physical RID on instance g          [E × n_e, int] │
    │ T: batch size, k: top-k, n_e: MoE instances                │
    └────────────────────────────────────────────────────────────┘

    ┌── AEBS CUDA Kernel (每个 MoE instance 独立执行) ──────────┐
    │ // Step 1: Collect activated expert set (GPU parallel)     │
    │ E_set = {}                                                 │
    │ Parallel for (i,j) in [0..T-1] × [0..k-1]:                │
    │     atomicOr(E_set_bitmap, L(i,j))  // 位图标记激活 expert │
    │                                                            │
    │ // Step 2: Initialize (per-instance state)                 │
    │ actRep[e] = -1 for all e in E_set  // selected replica    │
    │ load[g] = 0 for g = 1..n_e          // distinct expert cnt │
    │                                                            │
    │ // Step 3: Assign single-replica experts                   │
    │ for e in E_set where R(e) == 1:                           │
    │     g = unique instance in G(e)                            │
    │     actRep[e] = P(e,g)                                     │
    │     atomicAdd(load[g], 1)                                  │
    │                                                            │
    │ // Step 4: Assign multi-replica experts (greedy)           │
    │ for e in E_set where R(e) > 1:                            │
    │     g* = argmin_{g in G(e)} load[g]  // 最少负载实例      │
    │     actRep[e] = P(e, g*)                                   │
    │     atomicAdd(load[g*], 1)                                 │
    │                                                            │
    │ // Step 5: Rewrite token routing (GPU parallel)            │
    │ Parallel for (i,j) in [0..T-1] × [0..k-1]:                │
    │     O(i,j) = actRep[L(i,j)]  // 逻辑EID → 物理RID        │
    │                                                            │
    │ // 关键: 所有 MoE instances 用相同输入独立运行相同 kernel  │
    │ // 确定性算法保证一致性 → 无跨GPU协调                      │
    └────────────────────────────────────────────────────────────┘

    ┌── Dispatch (根据 O(i,j) 分发 token) ──────────────────────┐
    │ for each token i:                                          │
    │     for j in 1..k:                                         │
    │         dest_instance = instance_of(O(i,j))                │
    │         send activation[i] to dest_instance                │
    │         (通过 NVSHMEM one-sided put)                       │
    └────────────────────────────────────────────────────────────┘
    ```

  - **AEBS Scheduling Overhead 测量原理**：
    1. 在每个 MoE layer 的 forward path 中插入 CUDA event (cudaEventRecord) 测量 AEBS kernel 的 wall-clock 时间
    2. 变化 batch size (64, 128, 256, 512, 1024, 2048, 4096) 和 MoE instance 数 (8, 16)
    3. 关键结果：small batch (64) 下 <20μs，large batch (4096) 下 <90μs，始终远小于 MoE computation (~hundreds of μs)
    4. AEBS overhead 随 batch size 增长后趋于 plateau (因大部分 expert 已被激活)

  - **NVSHMEM Adaptive Two-Phase Communication 时序**：
    ```
    Case-1 (少量目标, 直接传输):
    Attention GPU A: [Aggregate intra-node via NCCL AllGather]
                   → [nvshmem_putmem_signal → MoE GPU E0, E1, ..., Em]
                   → [nvshmem_signal_wait ← MoE done]
    
    Case-2 (大量目标/数据, 中继传输):
    Attention GPU A: [Aggregate intra-node via NCCL]
                   → [nvshmem_putmem_signal → designated MoE relay GPU R]
    MoE relay R:     [nvshmem_signal_wait ← received]
                   → [intra-node NVLink multicast → local MoE instances]
    ```

## LongCat-Flash Technical Report

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - LongCat-Flash 在训练和推理侧均实现了大量定制 kernel：
    **训练侧 Kernel**：
    1. **Deterministic FlashAttention Gradient (FAG)**：默认 FAG 使用原子加法的非确定性归约。通过有限 extra workspace 以确定性顺序累积 tile，配合 double-buffer pipelining、tuned tiling schedules、load balancing 实现确定性和性能兼顾。性能达到原始确定性版本的 1.6x，非确定性版本的 0.95x。
    2. **Deterministic ScatterAdd**：默认实现因输入输出 operand count 不匹配，强制单 compute unit 串行执行导致 50x 减速。提出 hierarchical reduction algorithm，将所有可用 processors 并行化梯度聚合，性能与非确定性版本持平。
    3. **Optimized Grouped GEMM**：Grouped GEMM 计算量大但 compute density 低。三项优化：(a) Double-buffer pipelining 重叠计算、内存 I/O 和 epilogue；(b) Diagonal tiling 缓解 L2 cache 冲突；(c) HBM bandwidth control 通过限制 compute unit 使 Grouped GEMM 与 dispatch/combine 通信重叠。综合加速 5%-45%。
    4. **Fused GemmAdd**：dw 梯度累积时带宽受限。将 FP32 addition 融合到 GEMM epilogue 中，消除中间写回并在 tile GEMM pipeline 内完成加法。避免 BF16→HBM 转换精度损失，加速比 3.12x-3.86x。
    5. **IO-bound Kernel 重实现**：MoE layer permute/unpermute kernel 集成 drop-token 和 zero-computation experts 处理，保证确定性和性能。
    **推理侧 Kernel**：
    6. **MoE GEMM with SwapAB**：传统 MoE GEMM 以 token activations 为左矩阵（M×K）、expert weights 为右矩阵（K×N），M 维度需 64 元素最小对齐需 padding。SwapAB 反转映射——weights 为左矩阵、activations 为右矩阵——利用 N 维度的 8 元素粒度灵活填充，最大化 tensor core 利用率。
    7. **Custom Communication Kernels (NVLink Sharp)**：使用 PTX 内联汇编直接调用 NVLink Sharp 的 multimem.st（broadcast）和 multimem.ld_reduce（in-switch reduction），实现 reduce-scatter 和 all-gather。支持均匀和非均匀 token 分布，比 NCCL 和 MSCCL++ 更快（4KB-96MB message size 全范围），仅需 4 thread blocks。
  - 实验比较：
    - Deterministic FAG: 1.6x vs original deterministic, 0.95x vs non-deterministic
    - Deterministic ScatterAdd: 消除 50x 减速，达到与非确定性版本性能持平
    - Grouped GEMM: 5%-45% speedup over default
    - Fused GemmAdd: 3.12x-3.86x speedup
    - Communication Kernels: 比 NCCL 和 MSCCL++ 更快（4KB-96MB 全范围）
    - LongCat-Flash vs DeepSeek-V3 deployment performance (Table 6)

- 后端平台是什么，配置是什么。
  - **训练**：NVIDIA H800-80GB，tens of thousands accelerators，200Gb/s RDMA per accelerator，NVLink intra-node。
  - **推理**：NVIDIA H800-80GB (SXM5)，NVLink intra-node + RDMA inter-node (GPUDirect RDMA)。FlashMLA 可达 660 TFlops（H800 SXM5），DeepEP 带宽可达 40GB/s。FP8 量化推理支持。

- 评估性能的软件/脚本是什么。修改了什么。
  - 论文未明确提供具体的 kernel benchmark scripts 路径，但在架构部分详细描述了各 kernel 的设计和性能对比。
  - **训练侧**：kernel 集成在训练框架中。SDC 检测通过在 FlashAttention gradient backward 中嵌入 on-chip in-place recomputation 验证 bitwise 一致性。
  - **推理侧**：
    - DeepEP (https://github.com/deepseek-ai/DeepEP)：修改支持 zero-computation experts（zero-comp expert 输出无需通信）
    - EPLB：修改支持 zero-computation experts 的负载均衡
    - FlashMLA (https://github.com/deepseek-ai/FlashMLA)：用于 MLA kernel 性能参考
    - DeepGEMM (https://github.com/deepseek-ai/DeepGEMM)：用于 MoE GEMM kernel 性能参考
  - 修改/新增内容：
    1. SwapAB MoE GEMM kernel：权重/激活矩阵角色互换的 GEMM 实现
    2. NVLink Sharp PTX kernels：reduce-scatter 和 all-gather 的直接 PTX 实现
    3. TVD fused CUDA graph：将 Target forward + Verification + Draft forward 三个运算融合
    4. Multi-step overlapped scheduler 的 KV cache pre-allocation 逻辑

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - **开源情况**：模型和部分代码在 GitHub (https://github.com/meituan-longcat) 开源。使用的开源组件：FlashMLA、DeepEP（修改版）、DeepGEMM、NCCL、MSCCL++。
  - **Deterministic FAG kernel 原理**：
    ```
    [输入] Q, K, V, dO 及其梯度（BF16 tensors）
    [Kernel 执行流程]
    1. 将 dQ, dK, dV 计算按 tile 维度划分为多个 tile
    2. 使用 extra workspace 按确定性顺序累积各 tile 的梯度
       - 替代默认的 atomicAdd（非确定性归约顺序）
       - Workspace 存储各 tile 的部分结果
    3. Double-buffer pipelining: 当前 tile 计算与上一 tile 结果写入重叠
    4. Tuned tiling: 按 H800 SM 数量和 shared memory 大小优化 tile 尺寸
    5. Load balancing: 在各 SM 间均匀分配 tile 计算量
    [输出] 确定性的 dQ, dK, dV（bitwise 一致的梯度）
    [性能] 1.6x vs naive deterministic, 0.95x vs non-deterministic
    ```
  - **MoE GEMM with SwapAB 原理**：
    ```
    [传统 MoE GEMM] C = A × B
      - A: activations [m=token_count, k=expert_dim] → m 需 padding 到 64 对齐
      - B: weights [k=expert_dim, n=intermediate_dim]
    
    [SwapAB MoE GEMM] C^T = B^T × A^T
      - B^T: weights transpose [n=intermediate_dim, k=expert_dim] → 作为左矩阵
      - A^T: activations transpose [k=expert_dim, m=token_count] → 作为右矩阵
      - n 维度具有 8 元素对齐粒度（vs m 维度的 64 元素），padding overhead 显著降低
    
    [输入到输出]:
    Input: activations [m, k] (BF16/FP8) + weights [k, n] (BF16/FP8)
    → Swap: 内存 reinterpretation 替代物理转置
    → Tensor Core GEMM: 波前级别的 tile 并行
    → Output: [m, n] (BF16/FP8)
    ```
  - **NVLink Sharp Communication Kernel 原理**：
    ```
    [All-Gather via multimem.st]
    1. 每个 GPU 持有部分数据
    2. inline PTX: multimem.st 指令 → NVSwitch 硬件广播各 GPU 数据到所有参与者
    3. 结果: 所有 GPU 获得完整数据副本，仅 4 thread blocks 驱动
    
    [Reduce-Scatter via multimem.ld_reduce]
    1. 每个 GPU 持有完整数据的不同部分
    2. inline PTX: multimem.ld_reduce 指令 → NVSwitch 在交换过程中执行 in-switch reduction
    3. 结果: 各 GPU 获得规约后的分片数据
    [性能] 4KB-96MB message size 全范围超越 NCCL 和 MSCCL++
    ```

## LYNX: Enabling Efficient MoE Inference Through Dynamic Batch-Aware Expert Selection

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - LYNX 实现了 4 个 fused Triton CUDA kernels，将 MoE router 输出的后处理（confidence analysis, expert scoring, expert pruning, remapping）融合为高效 kernel，替代原本超过 700 个 PyTorch 小算子。具体 kernel 实现：
    1. **Kernel 1 — Token-wise Binning（逐 token 离散化）**：拦截 router logits，计算每 token 对 top-k 专家的 log-ratio（logit[e] - logit[top1]），做 AffinityBinning 离散化，同时计算 top-k 权重和。融合了原本需数百个 PyTorch element-wise ops（subtract, division, floor, clamp）。
    2. **Kernel 2-3 — Batch-wise Scoring & Expert Pruning（批次级评分与剪枝）**：以 batch_size 为底数的指数加权计算每个 expert 的 batch 级分数，基于分数分布和 bin width 动态确定 active expert set。融合了 reduce、scatter、top-k 选择等操作。
    3. **Kernel 4 — Expert Remapping & Compaction（专家重映射与压缩）**：将 low-confidence tokens 的 expert assignment 重映射到 active expert set，compaction 重排 token-to-expert 映射表，renormalize 权重并重新计算 top-k。融合了 gather、scatter、sort、softmax 等操作。
    4. 所有 kernel 保持静态控制流，支持 CUDA Graph capture。
  - 实验比较：
    - Latency breakdown：Baseline (vLLM default) vs LYNX 的端到端延迟分解为 expert computation 和 non-expert components
    - Kernel overhead：LYNX 的 4 个 fused kernel 开销 <4% 总体延迟
    - 不同 batch size（1-64）和 sequence length（512/4096）下的 TPOT
    - 与不同并行策略（TP, EP）的叠加效果

- 后端平台是什么，配置是什么。
  - **GPU**：NVIDIA H200 (141 GB HBM)，SXM NVLink
  - **CPU**：2x AMD EPYC 9554 64-Core，1.5 TB DRAM
  - **软件栈**：Ubuntu 22.04.4 LTS，CUDA 12.6，NVIDIA driver 560.35.05
  - **Kernel 框架**：Triton（4 个 fused kernels），PyTorch profiler 用于 kernel-level latency capture
  - **Offloading 实验**：NVIDIA A100，PCIe CPU-GPU 链路

- 评估性能的软件/脚本是什么。修改了什么。
  - vLLM v0.10.1 框架，PYTORCH CUDA profiler 捕获 kernel-level latency
  - 4 个 fused Triton kernel 是全新实现的，替代了 vLLM 默认 MoE router 后的 PyTorch dispatch pipeline
  - 修改内容：
    1. **新增 Confidence Analyzer Kernel**：Triton 实现，输入 router logits (B x N)，输出 per-token bin assignments 和 top-k weight sums。key operation：log_ratio discretization with α/β binning params。
    2. **新增 Adaptive Expert Scorer Kernels (x2)**：Triton 实现，输入 per-token bin assignments (B x k)，输出 batch-level expert scores (N) 和 active expert mask。key operation：exponential weighting with batch_size base, score-based thresholding。
    3. **新增 Expert Remapper Kernel**：Triton 实现，输入 active expert mask + per-token bin assignments，输出 compacted token-to-expert mapping + renormalized weights。key operation：gather-scatter remapping, softmax renormalization。
    4. **Phase-aware Optimizer in Batch Scheduler**：在 vLLM scheduler 中新增 memory-bound detection 逻辑（非 kernel，为 CPU 端调度逻辑）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - **开源情况**：论文未提供开源代码链接。实现基于 vLLM v0.10.1 + Triton。
  - **Kernel 执行全流程**（以 Qwen2-57B 单层 MoE layer, decode iteration, batch B=16, N=64, k=8 为例）：
    ```
    [输入] Router logits: B x N = 16 x 64 float tensor (GPU global memory)
    
    [Kernel 1 — Token-wise Binning]
    - Grid: (B, ) 即 16 thread blocks
    - 每个 block: 加载 1 个 token 的 router logits (64 elements) → registers
    - 计算: top-1 logit → log_ratio[e] = logit[e] - top1_logit for each e in top-k
    - 离散化: bin[e] = clamp(floor(log_ratio[e] * α), -β, 0)
    - 输出: bin assignments (B x k int) + top-k weight sums (B float) → global memory
    - 融合的 PyTorch ops: logit sort, subtraction, softmax (deferred), floor, clamp
    
    [Kernel 2-3 — Batch-wise Scoring & Pruning]
    - 输入: bin assignments (B x k), batch_size B
    - Grid: (N, ) 即 64 thread blocks (每个 expert 一个)
    - 每个 block: 遍历 batch 中所有 token，若 expert 在该 token 的 top-k 中则累加 B^{bin[token][expert]}
    - 计算: score[expert] = Σ_t B^{bin[t][expert]}
    - 阈值确定: 基于 score distribution + bin_width + max_bins 动态计算
    - 输出: active expert mask (N bool) → global memory
    
    [Kernel 4 — Expert Remapping & Compaction]
    - 输入: active expert mask, per-token bin assignments, top-k per token
    - 操作: 对每个 low-confidence token，将其 lower-ranked expert 重映射到 active expert set 中的替代专家
    - Compaction: 重排 token-to-expert 映射表为连续索引
    - Renormalize: 对 remapped assignment 重新计算 softmax → 最终 dispatch weights
    - 输出: compact mapping (B x k int), renormalized weights (B x k float)
    
    [后续] Expert GEMM kernel launch with reduced expert count
    - 原本需加载 ~25 个 experts 的权重 → 现在仅需加载 ~15-18 个
    - 从 HBM 读取量: (active_count / original_count) * expert_size 字节
    ```
  - **评估原理**：LYNX 的 kernel 在 expert computation 前执行，通过减少 active expert 数量来降低 HBM 带宽消耗。4 个 fused Triton kernel 的开销 (<4% 总体延迟) 远小于因减少 expert 加载而节省的内存带宽时间。用 PyTorch profiler 在每个 iteration 内按 kernel 分解延迟，测量 expert computation latency 的减少量与 LYNX kernel overhead 的差值作为 net gain。

## FineMoE: Fine-Grained Expert Offloading for Large Mixture-of-Experts Serving

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - FineMoE 的 kernel 调度/运行时计算实现：
    1. **C++ Expert Cache with CUDA Runtime API**：Expert management in GPU 使用 CUDA Runtime API 实现。Expert ID 通过 hash map 映射到不同 GPU devices，按 round-robin 分配以平衡 GPU 负载。GPU space 中的 task pool 使用异步线程调度和执行 expert prefetching 与 on-demand loading 任务。
    2. **异步 Publisher-Subscriber 架构**：将 Expert Map Searcher 的 map searching 和 expert prefetching 与 inference process 解耦。Expert Map Store 作为 message broker，inference process 持续 publish context 数据（semantic embeddings + expert probability distributions），Expert Map Searcher subscribe context 并异步 prefetch experts 到 Expert Cache。
    3. **Multi-GPU Expert Parallelism**：支持 multi-GPU inference with EP，experts 映射到不同 GPU devices 进行加载和 offloading。Expert 分配遵循 round-robin 均衡负载。
    4. **On-demand Expert Loading**：当 expert miss 发生时（gate network 指定的 expert 不在 GPU cache 中），FineMoE 暂停所有 expert prefetching 任务，立即从 CPU 加载缺失的 experts 到 GPU 内存以进行 fast serving。
    5. **Expert Eviction Priority**：基于 LFU + searched probability 的联合优先级：PRI^{evict}_{l,j} = 1 / (p_{l,j} * freq_{l,j})，低概率 + 低频使用 = 高 eviction 优先级。
  - 实验比较：
    - System overheads：图 17 展示 one iteration 的 latency breakdown——context collection、on-demand loading、异步的 map searching/prefetching/map update 各占多少
    - 结果：除异步操作外的总延迟 < 50ms（<1% iteration），可忽略不计
    - Ablation study on caching：FineMoE vs LRU vs LFU

- 后端平台是什么，配置是什么。
  - **RTX 3090 测试台**：6× NVIDIA GeForce RTX 3090 24GB, NVLink 互联, PCIe 4.0 32GB/s, AMD Ryzen Threadripper PRO 3955WX, 480GB CPU memory
  - **A100 测试台**：1× NVIDIA A100 80GB HBM2e, 2 TB/s 内存带宽
  - CUDA Runtime API 用于 expert management，多 GPU 间通过 EP (expert parallelism) 分布

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 **MoE-Infinity 代码库**（https://github.com/TorchMoE/MoE-Infinity）进行修改
  - Expert Cache 在 C++ 层修改 MoE-Infinity 的 expert management：
    1. 新增 async task pool 用于 prefetching 和 on-demand loading 任务的调度
    2. 修改 CUDA memory management 逻辑以支持 similarity-aware prefetching priority
    3. 修改 eviction 逻辑：从纯 LFU 改为 LFU + probability-based priority
  - 评估方法：测量 latency breakdown（图 17），TPOT under varying cache limits（图 12），prefetch distance sensitivity（图 15）

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - **开源情况**：基于 MoE-Infinity（https://github.com/TorchMoE/MoE-Infinity），FineMoE prototype 未发现独立开源仓库
  - **Expert Cache Kernel 执行全过程**（单 GPU, Mixtral-8×7B 推理为例）：
    1. **Expert Prefetching Task 入队**：Expert Map Searcher 确定 E_prefetch = {E_{l,j}}，计算每个 expert 的 prefetching priority = p_{l,j} / (l - l_now) → 按 priority 降序入队到 GPU task pool
    2. **Asynchronous Prefetching 执行**：GPU space 异步线程从 task pool 取最高 priority 任务 → CUDA Runtime API cudaMemcpyAsync(host_ptr, device_ptr, expert_size, cudaMemcpyHostToDevice, stream) → PCIe 4.0 32GB/s 传输 expert weights 从 CPU 到 GPU → 更新 Expert Cache（hash map expert_id → cached location）
    3. **Inference Forward Pass 同步执行**：推理进程持续执行 forward——每层 gate network 选 top-K experts → 查 Expert Cache hash map：若命中则 CUDA GEMM kernel 直接使用 GPU 上的 expert weights 计算 → 若 miss 则暂停 prefetching task pool，立即 cudaMemcpy 该 expert 从 CPU to GPU → 执行 forward
    4. **Expert Eviction**：当 Expert Cache 达到 GPU memory budget 时 → 遍历所有 cached experts → 计算 eviction_priority = 1/(p_{l,j} * freq_{l,j}) → cudaFree 释放最高 eviction priority（最不重要）的 expert 的 GPU memory → 可用空间继续容纳新 prefetch 的 experts
    5. **Performance Metric**：TPOT = (推理计算时间) + (expert miss count × T_e on-demand loading time)。expert hit rate = 1 - (expert miss count / total expert activations)。Latency breakdown 通过 profiling 各操作的 wall-clock time 获得
    6. **Evaluation 原理**：通过控制 Expert Cache GPU memory budget（6GB-96GB），测量不同 memory 约束下的 TPOT。FineMoE 在相同 GPU memory 下通过更精确的 expert prediction（higher hit rate = fewer on-demand loads）实现更低 TPOT

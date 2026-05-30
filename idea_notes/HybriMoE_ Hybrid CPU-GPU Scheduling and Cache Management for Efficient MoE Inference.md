## HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management for Efficient MoE Inference

- baseline方法是什么？
  - **kTransformers**：SOTA CPU-GPU hybrid MoE 推理框架。使用**静态映射策略**——基于历史 expert 激活频率，将高频激活 expert（如 shared expert）固定映射到 GPU，低频 expert 在 cache miss 时由 CPU 执行。缓存策略使用 LFU（Least Frequently Used）。全栈执行例子（以 Mixtral-8x7B on RTX A6000 + 10-core Xeon, single token decode 为例）：
    - **模型推理算法层**：标准 MoE decoder，token → Attention (GPU) → MoE Gate (GPU, top-K routing) → expert FFN。kTransformers 将高频 expert 固定映射到 GPU（hot expert cache），低频 expert 由 CPU 计算（cache miss 时）。无动态 workload 适应能力。
    - **系统框架层**：kTransformers (kernel injection + CPU-GPU parallel execution)。Warmup 阶段 profiling 确定高频 expert（基于 calibration data 上的历史激活频率）。Runtime 阶段：gate 输出后 → 检查每个 activated expert 是否在 GPU cache → hit: GPU 执行 Marlin 4-bit quantized GEMM → miss: CPU 执行 llama.cpp C++ GEMM → CPU→GPU copy output → 聚合。
    - **编译框架层**：论文未明确说明。PyTorch/C++，llama.cpp backend。
    - **kernel 调度层**：GPU 端使用 Marlin 4-bit quantized GEMM kernel，CPU 端使用 llama.cpp C++ GEMM kernel。CPU/GPU 执行在独立 stream/thread 上并行，但任务分配完全由预定义的 static mapping 决定——低频 expert 总是 CPU 执行，即使 GPU idle。LFU cache eviction 不考虑 expert routing score 的预测信号。
    - **硬件架构层**：NVIDIA RTX A6000 + Intel Xeon Gold 5220R (10 cores)，PCIe。CPU 首 expert 计算较慢（cold cache），但后续 expert 因 CPU cache 命中而加速。
  - Baseline 痛点：
    1. **静态映射不适应动态激活（核心痛点 1）**：MoE expert 激活模式高度不稳定——相比 neuron-level sparsity，expert 激活频率分布更均匀（图 3a），预测困难。kTransformers 的固定映射（基于历史频率）忽略 runtime 的动态 workload 变化，导致 suboptimal CPU/GPU 资源利用和 load imbalance（图 1b vs 1c）。
    2. **缺乏 MoE-specific 缓存策略（核心痛点 2）**：LFU/LRU 忽略 MoE expert routing score 的预测信号。图 3b 表明高 score expert 在下一 iteration 中更可能被重用，但 LFU 仅基于历史使用频率，无法利用 score 信息做前瞻性缓存决策。
    3. **预取缺乏优先级决策（核心痛点 3）**：现有预取方法未探讨当多个后续层 expert 可被预取时如何做优先级决策。在 MoE 中，由于残差连接导致的层间 hidden state 相似性，gating 信息可复用做预测，但预取哪个层的 expert 对整体调度效率影响最大未被讨论。
    4. **调度问题 NP-Hard 但存在可利用规律（核心痛点 4）**：CPU-GPU workload 分配是 NP-hard 问题，但在 MoE 特定上下文中，expert transfer time 恒定、GPU 计算时间与 expert 数线性、CPU 首 expert 慢后续快（cache 利用）等规律可被利用来设计高效调度规则，而现有方法未利用。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **HybriMoE 方法**：通过三个互补的优化设计，将 MoE CPU-GPU 推理的"静态映射"转变为"动态自适应调度"。
    1. **Dynamic Intra-Layer Hybrid Scheduling（解决痛点 1 和 4）**：
       - Baseline 缺陷：静态映射无法适应动态 workload 变化，NP-hard 调度问题未利用 MoE-specific 规律。
       - HybriMoE 方法：三条优先级规则（GPU 优先高负载缓存 expert、CPU 优先低负载未缓存 expert、PCIe 优先高负载未缓存 expert）将调度从排序问题简化为分配问题 `argmin max(CPU_TIME, GPU_TIME)`。执行前仿真（贪心 fill CPU/GPU/PCIe timelines）评估不同分配策略，runtime 选择最优。利用的关键规律：expert transfer time 恒定、GPU 时间 ∝ expert 数（nearly constant per expert）、CPU 首 expert 慢后续快（因 CPU cache 重用）。
    2. **Score-Aware Caching / MRS Policy（解决痛点 2）**：
       - Baseline 缺陷：LFU/LRU 忽略 expert routing score 的预测信号。
       - HybriMoE 方法：Minus Recent Score (MRS) 策略，S = α × TopP(s) + (1-α) × S。仅累积 top-p 个 expert 的 score（因低 score expert 的 reuse probability 差异不显著）。利用图 3b 的观察——高 score expert 重用概率显著更高。25% cache capacity 下 MRS 比 LRU 高 6-8% hit rate。
    3. **Impact-Driven Prefetching（解决痛点 3）**：
       - Baseline 缺陷：预取缺乏多候选下的优先级决策机制。
       - HybriMoE 方法：复用后续 3 层的 gating 信息预测 expert activation → 对每个候选 expert 仿真其预取对整体调度效率的影响（复用 IV-B 的仿真逻辑）→ 贪心选择 impact 最高的 expert 预取。低开销（仿真 <μs 级），适合 real-time 推理。
  - 全栈执行例子（HybriMoE on RTX A6000 + 10-core Xeon, Mixtral-8x7B single token decode, 25% GPU cache ratio）：
    - **模型推理算法层**：与 baseline 相同的 MoE 模型结构（Mixtral-8x7B, top-2 routing）。差异在于运行时行为：
      - Warmup: profiling CPU/GPU expert latency + PCIe transfer latency → 初始化 MRS cache (random k experts)。
      - Per-layer: Gate → top-K expert selection → Simulation Scheduler (优先级队列 + 贪心 fill timelines) → 输出 expert-to-device assignment → Multi-stream parallel execution (GPU Marlin 4-bit GEMM + PCIe cudaMemcpyAsync + CPU llama.cpp C++ GEMM) → Impact-driven prefetching (预测 next-3-layers expert activation → 仿真评估预取收益 → 贪心预取) → MRS cache update (S = α·TopP(s) + (1-α)·S)。
    - **系统框架层**：基于 kTransformers + llama.cpp kernels。核心修改：Hybrid Scheduler (优先级规则 + simulation) 替换 static mapping；MRS Cache Manager 替换 LFU；Prefetching Module (impact-driven) 新增。Parallel execution engine: fine-grained CUDA stream (GPU compute + PCIe transfer) + CPU thread pool，CUDA event sync。
    - **编译框架层**：论文未明确说明。PyTorch/C++ (llama.cpp backend)。
    - **kernel 调度层**：
      - GPU Stream 0: Marlin 4-bit quantized GEMM kernel (SOTA 4-bit GPU GEMM) for cached experts。
      - GPU Stream 1: cudaMemcpyAsync for PCIe expert weight transfer (CPU→GPU or GPU→CPU)。
      - CPU Thread Pool: llama.cpp C++ GEMM kernel for uncached experts (CPU 端，首 expert cold cache → 后续 expert cache 命中加速)。
      - Simulation Scheduler: CPU 侧轻量级仿真 (<μs 级 latency)，不占 critical path。
      - 关键时序（以 4 experts: E₁(cached,high), E₂(uncached,high), E₃(cached,low), E₄(uncached,low) 为例）：
        ```
        Time →
        GPU:     |=== E₁ (Marlin) ===|=== E₂ (transferred+execute) ===|
        PCIe:    |== E₂ transfer ==|                                   |
        CPU:     |== E₄ (low-load) ==|                                 |
        ```
        vs Baseline (static mapping, E₂ 固定 CPU):
        ```
        GPU:     |=== E₁ ===|=== E₃ ===|  // idle after
        CPU:     |==== E₂ (high-load) ==============|== E₄ ==|
                 ↑ CPU 成为瓶颈
        ```
    - **硬件架构层**：与 baseline 相同（RTX A6000 + Xeon Gold 5220R 10-core）。但硬件利用率显著提升：GPU 避免了 baseline 中因等待 CPU 完成 heavy expert 导致的 idle，CPU 专注于其擅长的 low-load expert（延迟与 load 线性相关，low-load 时 CPU 优势最大）。
    - **关键性能数据**：
      | Model | Stage | Cache | HybriMoE vs kTransformers |
      |-------|-------|-------|---------------------------|
      | All (avg) | Prefill | 25-75% | 1.33× |
      | All (avg) | Decode | 25-75% | 1.70× |
    - **核心设计洞察**：HybriMoE 的本质洞察是将 MoE CPU-GPU 的 "task-to-hardware mapping" 问题解耦为三个子问题——(i) 当前层的 expert 如何分配到 CPU/GPU（intra-layer scheduling），(ii) 下一层哪些 expert 值得预取（inter-layer prefetching），(iii) 跨 iteration 哪些 expert 值得缓存（inter-iteration caching）。这三个子问题的时间维度不同（intra-layer 是 ms 级优化，inter-layer 是 μs 级预测，inter-iteration 是 s 级状态管理），HybriMoE 对每个时间维度使用了匹配的优化机制。优先级规则的设计体现了对 MoE CPU-GPU 异构计算特性的深刻理解——GPU 延迟近乎恒定（memory-bound），CPU 延迟线性增长（compute-bound），PCIe 传输恒定——这三者的"恒定 vs 线性"差异正是动态调度的 physics 基础。MRS 的 elegant 之处在于它不需要额外的预测模型（如 MLP predictor），而是直接利用 gating network 天然输出的 routing score——因为这个 score 本身就编码了"模型认为哪些 expert 对当前 token 重要"的信息，而"重要的 expert 更可能被未来 token 重用"是 MoE expert specialization 的自然推论。

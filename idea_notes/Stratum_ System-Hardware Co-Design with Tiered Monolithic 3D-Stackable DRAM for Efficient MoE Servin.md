## Stratum: System-Hardware Co-Design with Tiered Monolithic 3D-Stackable DRAM for Efficient MoE Serving

- baseline方法是什么？
  MoE 模型（如 Mixtral 8×7B、OLMoE、Llama-4-Scout）使用 HBM + GPU 的标准 serving pipeline：
  - **算法层**：Standard top-K gating (e.g., top-2)，所有 experts 在 GPU HBM 中均匀存储，无 topic-aware placement。Router 计算在 GPU 上轻量执行，expert FFN 在 GPU SM 上通过 Grouped GEMM 计算。
  - **系统框架层**：vLLM（或类似框架）做 continuous batching，请求按 FIFO 或 priority 调度，不考虑 query topic 与 expert affinity。所有 MoE expert weights 常驻 GPU HBM，decode 阶段注意力机制因 KV cache 访问成为 memory-bound bottleneck。
  - **编译框架层**：论文未明确说明，标准 PyTorch → CUDA kernel 编译路径。
  - **Kernel 调度层**：GPU SM 执行 Grouped GEMM（fused MoE kernel），每个 expert 的 tokens 通过 all-to-all dispatch/combine 在 GPU 内 HBM 上读写。Attention 的 KV cache 访问受限于 HBM bandwidth（~800 GB/s per stack），decode 阶段因批量小成为 memory-bound。
  - **硬件架构层**：NVIDIA H100/A100 GPU + HBM3 stack。HBM 通过 TSV（10μm pitch）连接 DRAM dies stack 和 base die，经由 1024-bit I/O 和 silicon interposer 与 GPU 通信。HBM 内部带宽受限于 TSV 数量，外部 bandwidth 受限于 interposer I/O。DRAM 工艺针对存储优化、不擅长逻辑计算，导致 NMP 在 DRAM die 内实现计算会面临 PPA 开销和散热挑战。
  - **芯片设计层**：HBM 通过 die stacking + TSV 互联，每 stack 6-12 层 DRAM dies，base die 通过 interposer 连接 GPU。制造 yield 低、成本高（TSV fabrication + bonding）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  Stratum 通过 **系统-硬件协同设计（System-Hardware Co-Design）**，在三层（算法/系统、kernel/计算映射、硬件/芯片）同时对 baseline 进行改进：
  - **算法/系统层（Topic-Aware Expert Prediction + Placement）**：
    - Baseline 缺陷：所有 experts 在 HBM 中均匀存储，heat map 不考虑查询语义。热门 expert 和冷门 expert 访问延迟相同，内存带宽浪费在低效访问上。
    - Stratum 方法：(a) 用 DistillBERT-based topic classifier（67M params）预测查询主题 → SLO-aware scheduler 将同 topic 请求 batch 在一起 → 利用 offline profiled per-topic expert hit rate 表预测 batch 内 experts 的使用概率 → Algorithm 1 将 hot experts 放入 Mono3D DRAM 快 tier（tRCD=2.29ns）、cold experts 放入慢 tier（tRCD=22.88ns）；(b) 当切换 topic batch 时，通过 NMP 的 row-swap buffer 在 DRAM bank 内执行 expert swap，避免 traversing 高延迟的 interposer path（overhead <0.37% time, <0.03‰ energy）。
  - **Kernel/计算映射层（NMP Operator Mapping + Pipeline Optimization）**：
    - Baseline 缺陷：decode 阶段的 attention 和 MLP 因 HBM 带宽限制成为 memory-bound（GPU 计算资源利用率低）。Expert 之间无执行调度优化。
    - Stratum 方法：(a) Expert Processing——tensor parallelism 分区策略（GeMM1/2 垂直分片、GeMM3 水平分片），避免 expert weight 跨 PU 复制。所有 PUs 协作顺序执行 expert，输入 token 通过 sub-ring all-gather 复制。(b) Pipeline 优化——GeMM2 与 activation 重叠、GeMM3 reduce-scatter 与下一 expert GeMM1 重叠、weighted-sum 立即执行、输入 token 分片发送到各 DRAM channel 减少传输延迟。(c) Attention Processing——head-level parallelism 将 heads 分配到 PU groups，interleaved Softmax/MatMul pipeline 减少 latency。K/V 沿 sequence length 分片，利用 ring network 的标量交换进行全局 Softmax 归一化。
  - **硬件/芯片层（Mono3D DRAM + Hybrid Bonding + Logic Die NMP）**：
    - Baseline 缺陷：HBM 内部带宽受 TSV 数量限制（10μm pitch），外部 bandwidth 受 interposer I/O 限制。DRAM die 嵌入逻辑计算面临 PPA 开销大、散热差的问题。
    - Stratum 方法：(a) Mono3D DRAM 替代 HBM——1024 层垂直 stackable 1T1C DRAM，Cu-Cu hybrid bonding（1μm pitch, 5× denser than HBM TSV），内部带宽 19-34 TB/s（远超 HBM ~800 GB/s）。(b) Logic Die NMP——7nm 专用逻辑 die 通过 hybrid bonding 与 Mono3D DRAM die 直接互联，避免 DRAM process constraint 和 TSV 带宽瓶颈。128 TFLOPS peak compute, 16 PUs, on-chip ring network。(c) In-Memory Tiering——利用 Mono3D DRAM 的 WL staircase 延迟变化（tRCD 2.29-22.88ns），定义 8 个 memory tiers，通过 per-PE tiering table 动态控制 tRCD，快 tier 1.6× faster than slow tier。(d) CMOS-Under-Array (CUA)——高电压 DRAM 外围电路在 32nm CUA 层实现，低电压逻辑在 7nm logic die 实现，thermal modeling 确保 45W/chip 功率预算（vapor chamber + liquid cooling）。
  - **全栈执行例子（对比 Baseline）**：
    - Baseline（vLLM + H100, Mixtral 8×7B, decode one token）：embedding → attention（Q@K^T from HBM KV cache, ~800 GB/s bottleneck）→ gate routing → expert selection → load expert weights from HBM → GPU SM Grouped GEMM → weighted sum → output projection → next token。整个 decode loop 受 HBM bandwidth 约束（~800 GB/s per stack, ~3.35 TB/s 8 stacks aggregate）。
    - Stratum（Mono3D DRAM NMP, Mixtral 8×7B, decode batch）：xPU routing → prefill on xPU → send tokens + routing to Mono3D DRAM → Topic classifier predicts topic batch → Algorithm 1 places hot experts in fast tier → NMP 执行 expert computation with 19-34 TB/s internal bandwidth → ring network all-gather input tokens → sequential expert GEMM with pipeline overlap → attention on NMP with head-level parallelism → KV cache in intermediate tier → output tokens → xPU retrieve。整个流程的 memory 访问在 Mono3D DRAM 内部以 19-34 TB/s 带宽完成，仅 token I/O traverses interposer（1024-bit @ 6.4 Gbps = 819 GB/s），但大部分计算数据在 Mono3D DRAM-NMP 内部闭环，实现 8.29× throughput 和 7.66× energy efficiency 提升。
  - **关键设计决策映射**：
    | Baseline 缺陷 | Stratum 设计 | 层次 |
    |---|---|---|
    | Expert 访问延迟统一，无语义优化 | Topic classifier + hot/cold tier placement | 算法/系统 |
    | HBM 带宽不足（decode memory-bound） | Mono3D DRAM internal bandwidth 19-34 TB/s | 芯片设计 |
    | TSV 限制内部带宽 | Hybrid bonding 1μm pitch, 5× denser | 芯片设计 |
    | DRAM die 内计算 PPA 差 | 7nm logic die (128 TFLOPS) | 硬件架构 |
    | Expert swap 需 travers GPU-HBM | Near-memory row-swap buffer | Kernel调度 |
    | GPU SM 计算-通信串行 | Ring network + pipeline overlap | Kernel调度 |
    | 无 memory tier 概念 | 8-tier Mono3D DRAM (tRCD 2.29-22.88ns) | 硬件架构 |

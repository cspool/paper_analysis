## Bit-Width-Aware I/O-Compute Pipeline (位宽感知 I/O-计算流水线)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Bit-Width-Aware I/O-Compute Pipeline 是 D2MoE 提出的端侧 MoE 推理流水线范式，在细粒度 bit-width 级别重新组织 expert 权重的 I/O 加载和计算执行顺序，以最小化 I/O 等待造成的计算空闲气泡（idle bubble）。

核心创新：传统的 I/O-Compute 按 expert ID 顺序依次加载和计算（某个 expert 计算时后续 expert 的 I/O 尚未开始，形成 bubble）。D2MoE 利用 MWQ 的嵌套特性——高 bit-width 权重包含低 bit-width——在 bit-width 级别重组队列。例如 3 个 request 选 Expert 2（1 个选 INT2、2 个选 INT3），传统方式需加载 1 份独立 INT2 + 2 份独立 INT3；MWQ 方式仅加载 1 份 shared INT2 base + 2 份 1-bit INT3 residual，且 INT2 base 加载时其它 residual I/O 可与 computation 重叠。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
以 Environment 1 (RTX 3060, 6GB), LLaMA-MoE-3.5B, M=1600MB, 32 requests 为例：

```
=== Bit-Width-Aware Pipeline 执行流程 ===

Phase 1: 离线 Profiling（安装时执行一次）
    for each bitwidth b_k in {b_1, ..., b_K}:
        T_io(b_k)  = measure disk→GPU transfer latency (平均)
        T_comp(b_k) = measure dequantization + GEMM latency (平均)
    
Phase 2: 在线推理（每 layer 每 token）
    Input: 当前 layer 各 expert 的 bit-width 选择矩阵 B[j][k]
           (B[j][k] = 第 j 个 expert 的第 k 种 bit-width 被选中的次数)
    
    1. Memory Budget Scheduler (Algorithm 2):
       if current_layer_memory > M:
           优先释放高 bit-width residual weights（保留 base 低 bit-width）
           if still > M: 释放低 bit-width base weights
    
    2. HEBF 调度:
       for each expert j:
           Q_j = [b_1, b_2, ..., b_K]  (按 bit-width 升序排列)
       while Q is not empty:
           pop 所有 expert queue 的 head element 中 B[j][k] 最大的
           enqueue to I/O Queue: 加载该 bit-width expert weight
   
    3. Parallel Execution:
       I/O Stream:      cudaMemcpyAsync(disk→GPU Global Memory)
       Compute Stream:  反量化 (CUDA cores) → GEMM (Tensor cores)
       
       Timeline (以 Expert 2 为例, 1×INT2 + 2×INT3):
       I/O:    |-- INT2 Load --|--- 2×INT3 Residual Load --|
       Comp:                   |-- INT2 Deq+GEMM -----|-- 2×INT3 Deq+GEMM --|
       Bubble: # (几乎消除)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
D2MoE 使用 PyTorch + Triton 实现 I/O-Compute 并行，CUDA kernel 基于 Ampere/Ada Lovelace 架构。使用独立的 CUDA stream 管理 I/O (cudaMemcpyAsync) 和 Computation (dequantization + GEMM)，通过 Triton 协调 pipeline 执行。目标是使得不等式 ∥L(s+1, j, k) ≤ C(s, j, k)∥（加载完成后才开始计算）和 ∥L(s, j, k) ≤ L(s, j, k+1)∥（按 bit-width 升序加载以最大化低 bit-width 复用）同时满足。

对比四种 pipeline 范式：
- (a) 传统：按 expert ID + bit-width 升序，无 nesting，大 bubble
- (b) MWQ without reorder：嵌套存储但保持顺序执行，减少 I/O 量但 bubble 仍显著
- (c) Fine-grained reorder：bit-width 级重排但无最优调度，bubble 减小
- (d) D2MoE (HEBF)：bit-width 级重排 + 频率优先，几乎消除 bubble

涉及论文标题：
- D2MoE: Dual Routing and Dynamic Scheduling for Efficient On-Device MoE-based LLM Serving

## Expert Processing with Tensor Parallelism on NMP (NMP上的张量并行专家处理)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Processing with Tensor Parallelism on NMP 是 Stratum 中将 MoE expert 计算映射到 NMP logic die 多个 PU 上并行执行的核心调度策略。MoE 的每个 expert 执行三个级联 GeMM 操作（projection-up GeMM1: W_1 ∈ R^{K×N}, projection-up GeMM2: W_2 ∈ R^{K×N}, projection-down GeMM3: W_3 ∈ R^{N×K}），Stratum 采用 tensor parallelism 将每个 expert 的权重矩阵分片到所有 PU 上，所有 PU 协作处理一个 expert（sequential across experts），而非并行处理多个 experts。矩阵分区策略：(a) GeMM1/2 沿 N 维（列）垂直分片——W_1[i] ∈ R^{K×(N/P)}；(b) GeMM3 沿 K 维（行）水平分片——W_3[i] ∈ R^{(N/P)×K}。这种分区避免了 expert weight 的跨 PU 复制（因不沿 M 维分片），同时消除了 GeMM2→GeMM3 之间的跨 PU 通信（每 PU 已拥有所需的 W_3 slice 和对应的输入 slice）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# MoE Layer Execution on Stratum NMP (16 PUs)
Input: X_t [M×K] (tokens in batch), expert routing IDs, gating weights w_e
Output: MoE layer output [M×K]

# Step 1: xPU sends X_t to Mono3D DRAM, switches to NMP mode
# Step 2: Sub-ring all-gather X_t to all 16 PUs

For each activated expert e (sequential):
  # Step 3-4: GeMM1 + GeMM2 (parallel on all PUs)
  For PU_i in [0..15] (parallel):
    Z_1[i] = X_t @ W_1[i]    # [M × N/P], PE Tensor Core 16×16 MAC
    Z_2[i] = X_t @ W_2[i]    # [M × N/P], parallel with Step 5
  
  # Step 5: Activation (overlapped with GeMM2)
  For PU_i in [0..15] (parallel):
    A[i] = SiLU(Z_1[i])              # Special Function Engine
    X_2[i] = A[i] ⊙ Z_2[i]           # Hadamard, no inter-PU comm needed
  
  # Step 6: GeMM3 (parallel on all PUs)
  For PU_i in [0..15] (parallel):
    Z_3[i] = X_2[i] @ W_3[i]   # [M × K/P], PE Tensor Core
  
  # Step 7: Reduce-scatter Z_3 across PUs via ring network
  # Each PU_i gets slice of final output
  # Overlapped with next expert's GeMM1 (pipeline)

# Step 8: Weighted sum of expert outputs
For PU_i in [0..15] (parallel):
  Y += w_e * Z_3_concat[i]   # Special Function Engine, on-the-fly

# Step 9: Write back to DRAM → exit NMP mode → xPU reads
```

关键 Pipeline 优化：
- GeMM2 || Activation（无数据依赖，并行执行）
- Reduce-Scatter(Expert N) || GeMM1(Expert N+1)（通信与计算 overlap）
- Weighted-Sum 在 expert 输出就绪后立即执行（minimize idle cycles）
- Input token 分片发送到各 DRAM channel → sub-ring all-gather（减少传输延迟）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Stratum 的张量并行专家处理的关键实现考量：(1) Sequential expert execution（非 parallel across experts）——因不同 experts 的 token count 不同会导致 PU 间负载不均，sequential 利用所有 PU 协作处理每个 expert 保证负载均衡；(2) Intra-PU matrix partitioning——PE 间主要按权重矩阵长边分片以最大化 tensor core utilization；(3) Communication-computation overlap——ring network 的 reduce-scatter 延迟被下一 expert 的计算完全隐藏（前提是 reduce-scatter latency ≤ GeMM1 latency）；(4) 输入 token 复制成本——all-gather X_t 到所有 PU 的 cost 被 amortize（因所有 activated experts 共用同一 X_t）。Cycle-level simulation 验证这些 optimizations 使 NMP 的 expert processing 未因为 sequential 而成为瓶颈。

涉及论文标题：
- Stratum: System-Hardware Co-Design with Tiered Monolithic 3D-Stackable DRAM for Efficient MoE Serving

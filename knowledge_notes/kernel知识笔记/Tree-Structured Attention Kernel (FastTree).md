## Tree-Structured Attention Kernel (FastTree)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Tree-Structured Attention Kernel 是 FastTree 提出的 GPU attention kernel，专门为 radix tree 组织的 KV cache 设计。其核心思想是：利用 radix tree 的 KV cache 共享结构，将共享同一 context prefix 的 queries 聚合为 context-queries groups，在单个 kernel launch 中并行处理所有 groups，替代传统 per-query 分离计算的 attention kernel。Kernel 以 FlashAttention 的 tiled online softmax 风格执行：Q tile 沿 query dimension 并行化（跨 GPU thread blocks），KV tile 在 context dimension 串行迭代（block 内循环）。关键优化：(1) query aggregation 使 Q 从 vector 变为 matrix，GEMV→GEMM 使能 tensor core；(2) shared memory 内的 KV tile 被同一 group 内所有 queries 复用，消除 HBM 重复加载；(3) 单 kernel 处理所有 groups，消除 per-query kernel launch overhead。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FastTree attention kernel 的执行流程（以 Triton 实现为基础，3-level tree, H100 GPU）：

```
// === Step 0: Input ===
Input: grouping_plan = [(ctx_1, {q_1,...,q_m}), (ctx_2, {q_m+1,...,q_n}), ...]
       // 每个 group = (shared context prefix, aggregated queries)

// === Step 1: Single kernel launch, process all groups ===
for each group g = (ctx_g, Q_g) in parallel (across block sets):
    // Q_g: queries aggregated for this context
    // ctx_g: shared KV cache context for this group

    // Tile Q matrix along query dim for parallelization
    Q_tiles = tile(Q_g, dim=query, tile_size=T_q)

    for each Q_tile in parallel (across thread blocks):
        // Initialize online softmax state
        O_partial = zeros(T_q, d)       // partial output
        L_partial = zeros(T_q, 1)       // LogSumExp accumulator
        m_partial = -inf * ones(T_q, 1) // running max

        // Iterate KV tiles along context dim (sequential within block)
        KV_tiles = tile(ctx_g, dim=context, tile_size=T_c)
        for K_tile, V_tile in KV_tiles:
            // Load from HBM → shared memory
            Q_smem = load(Q_tile)         // from HBM
            K_smem = load(K_tile)         // from HBM (once per group!)

            // BMM1 on tensor core: S = Q @ K^T
            S = matmul(Q_smem, K_smem^T) / sqrt(d)   // GEMM, not GEMV!

            // Online softmax update
            m_new = rowmax(S)
            m_updated = max(m_partial, m_new)
            L_new = exp(m_partial - m_updated) * L_partial
                  + rowsum(exp(S - m_updated))
            O_partial = exp(m_partial - m_updated) * O_partial

            // BMM2 on tensor core: P @ V
            P = exp(S - m_updated)
            V_smem = load(V_tile)         // once per group
            O_partial += matmul(P, V_smem)

            m_partial = m_updated
            L_partial = L_new

        // Write partial results to HBM
        store(O_partial, L_partial)

// === Step 2: Lightweight reduce kernel ===
for each query q:
    O_final[q] = 0
    L_total = 0
    for each group g where q participates:
        O_final[q] += O_partial[q][g] * L_partial[q][g]
        L_total += L_partial[q][g]
    O_final[q] /= L_total
```

与 baseline（per-query FlashAttention）的关键差异：
```
FlashAttention decode (per-query):
  for each query q:           ← N kernel launches (or N blocks, separate)
      for KV tile in ctx_q:   ← ctx_q loaded from HBM for EACH query
          GEMV: Q·K^T         ← Q is vector (1×d), matrix-vector
          softmax
          GEMV: P·V           ← P is vector (1×T_c)

FastTree (grouped):
  for each group g:           ← fewer kernel launches
      for KV tile in ctx_g:   ← ctx_g loaded from HBM ONCE per group
          GEMM: Q·K^T         ← Q is matrix (T_q×d), matrix-matrix, tensor core
          softmax
          GEMM: P·V           ← P is matrix (T_q×T_c), tensor core
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FastTree 的 attention kernel 使用 Triton 实现（Python DSL, version 3.0.0），不依赖 Hopper 特有特性（如 TMA），具有较好的跨 GPU 可移植性。Kernel 集成到 SGLang v0.2.13 的 attention backend，decode 阶段使用 FastTree kernel，prefill 阶段沿用 FlashInfer。Multi-phase tiling 通过编译多个 tile size 的 kernel 变体实现——在 tree 不同层级（root 聚合 query 多 → 大 tile；leaf query 少 → 小 tile）动态选择。Reduce kernel 融合 LogSumExp rescaling（标准 FlashAttention reduction pattern）。开源地址：https://github.com/PanZaifeng/FastTree-Artifact（Apache-2.0）。在 H100 上 kernel benchmark 显示平均 5.1× over FlashAttention, 4.2× over FlashInfer, 10.6× over DeFT。

涉及论文标题：
- FastTree Optimizing Attention Kernel and Runtime for Tree-Structured LLM Inference

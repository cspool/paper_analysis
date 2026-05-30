## Loop Schedule (Spatial/Temporal Partitioning for GEMM Chain Fusion)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Loop Schedule 是 FlashFuser 中定义 fused GEMM chain kernel 的 loop 执行顺序和维度划分的策略。给定算子链的统一 loop 维度集合 X = {M, N, K, L}，Loop Schedule 将其分为 Spatial dimensions (S, 多个 SM 并行计算) 和 Temporal dimensions (T, 单 SM 串行计算)，并定义 permutation 确定 nesting order。共 41 种可能组合（|S|=1: 24种, |S|=2: 12种, |S|=3: 4种, |S|=4: 1种）。不同的 loop schedule 影响中间 tensor 的缓存需求和 data movement volume。

从kernel调度角度拆解术语：
```
// 相同 GEMM chain 在不同 loop schedule 下的差异:

// MLNK order (K,L outer→inner):
// 需要存储完整 C [blk_m × blk_n] tile 在片上
for m in M_tiles:
  for l in L_tiles:
    for n in N_tiles:
      for k in K_tiles:
        C_local[m,n] += A[m,k] × B[k,n]
    // C_local 完整 → 需 spilling to DSM if > SMEM
    for k in K_tiles:
      E_local[m,l] += C_local[m,:] × D[:,l]

// MNLK order (K inner, L before K):
// 每次 LK iteration 产生 partial E, 更高效的 register accumulation
for m in M_tiles:
  for n in N_tiles:
    for l in L_tiles:
      for k in K_tiles:
        // interleaved: compute C partial → immediately consume for E
        C_partial[m,n] += A[m,k] × B[k,n]
        E_partial[m,l] += C_partial[m,:] × D[n,l]
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FlashFuser 的 Dataflow Analyzer (Algorithm 1) 接受 LoopSchedule s 作为输入，对每个 candidate 分析：(1) Spatial dims 不串行遍历——其 effective size 设为 tile size（由 cluster-level tile 处理）；(2) Temporal dims 通过反序遍历确定各 tensor 跨 tile 的重复访问次数；(3) 联合 tile size 和 resource mapping 生成具体 spill plan。Search Engine 枚举 41 种 LoopSchedule 候选用 cost model 选 minmax data movement 的最优配置。

涉及论文标题：
- FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection

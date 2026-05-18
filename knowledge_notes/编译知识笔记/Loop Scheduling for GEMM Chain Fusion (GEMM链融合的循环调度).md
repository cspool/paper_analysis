## Loop Scheduling for GEMM Chain Fusion (GEMM链融合的循环调度)

术语是什么？通过联网搜索让回答具体和精准。

Loop Scheduling for GEMM Chain Fusion 是 FlashFuser 编译框架中用于融合多个 GEMM 算子时决定循环执行顺序的技术。当多个 GEMM 算子连续排列时（如 FFN 中的两个 GEMM），每个算子有自己的 loop dimensions (M, N, K, L)，融合后的 kernel 需要将这些独立维度统一为 interdimensional set X = {x0, x1, ..., xJ-1}。Loop Schedule 定义：(1) permutation s——决定嵌套循环中这些独立维度的遍历顺序（如 MNKL, MNLK, MLNK 等）；(2) 维度 partition——将每个维度标记为 spatial（S，由多个并行处理单元同时计算）或 temporal（T，由单个处理单元顺序计算）。对 4 维问题，spatial dimensions 数从 0 到 4，对应 schedule 数从 1（全 temporal）到 41（1 spatial: 24 种 + 2 spatial: 12 种 + 3 spatial: 4 种 + 4 spatial: 1 种）。不同 loop schedule 影响需要 cached 的中间 tensor 大小。

从编译框架角度拆解术语：

FlashFuser 中 Loop Schedule 对中间 tensor 大小的影响（以 MLNK vs MNLK 为例）：

```
// MLNK Order (M→L→N→K):
for m in M_blocks:
  for l in L_blocks:
    for n in N_blocks:
      for k in K_blocks:
        C[m,n] += A[m,k] × B[k,n]  // GEMM0: 累加到 C
        E[m,l] += C[m,n] × D[n,l]  // GEMM1: 立即消费 C 的当前行
      // C[m][*] 行在使用后即可释放，不需要完整缓存
      // 仅需缓存 C 的当前 n tile

// MNLK Order (M→N→L→K):
for m in M_blocks:
  for n in N_blocks:
    for l in L_blocks:
      for k in K_blocks:
        C[m,n] += A[m,k] × B[k,n]  // GEMM0: 计算 C[m,n]
      // C[m,n] 完全累加完成后才被后续 L 循环消费
      // 需要本地存储完整 C tensor，spilling 需求更大
```

FlashFuser 的 Dataflow Analyzer 根据 loop schedule 和 tile size 确定每种顺序下 reused tensor 的 footprint，贪心分配到 reg→SMEM→DSM→global 层次。cost model 以 C_l = V_l/B_l 评估每层数据搬移成本，minimax 优化选择最佳 loop schedule。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 FlashFuser 的 Fusion Search Engine 中，loop scheduling 作为搜索空间的第一维度被枚举：
1. **候选生成**：对所有 J 个 loop dimensions 枚举 spatial dimensions 数（0 到 J），对每个 spatial count 枚举 spatial dimension 组合和 temporal dimension 排列（共 41 种 schedule）
2. **Pruning**：Rule 3（activation constraint）要求前一个 GEMM 的 accumulation dimension 必须位于 innermost loop——否则会产生无法被激活函数消费的 partial sum
3. **Cost evaluation**：Dataflow Analyzer 对每种 schedule 计算 data movement volume，cost model 给出估计代价
4. **Selection**：Top-K profiling 后选择实测最优的 schedule

与 previous work 的差异：Chimera 也考虑 block execution order，但限制在单 SM SMEM 中，不考虑 DSM。FlashFuser 将 loop schedule 扩展到多层 memory hierarchy 的分析。

涉及论文标题：
- FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection

## Fusion Search Engine with DSM-aware Pruning（DSM感知的融合搜索引擎）

术语是什么？通过联网搜索让回答具体和精准。

Fusion Search Engine 是 FlashFuser 编译框架的前端组件，负责在极大搜索空间中高效找到最优 fusion execution plan。搜索空间由 LoopSchedule（41 种）、TilingSize（block-level tile + cluster-level tile）、ResourceMapping（tensor placement）三维变量组成。以 GPT-6.7B FFN (M=256, N=16384, K=4096, L=4096) 为例，初始搜索空间约 2.75 × 10^13 个候选。与 previous work (Chimera, 搜索空间约 11,550 ≈ 10^4) 的关键差异：DSM 的引入使更多 fusion 策略可行，搜索空间扩大约 10^9 倍。Search Engine 使用 5 条 pruning rules 将搜索空间降至约 1.15 × 10^6，再用分析性 cost model 筛选 top-K (K=11) 候选，最后硬件 profiling 选最优。

从编译框架角度拆解术语：

```
Fusion Search Engine 搜索流程 (Algorithm 2):

all_candidates = EnumerateAllCandidates(g, d)
// 初始空间: 41 schedules × 54 cluster configs × tile choices ≈ 2.75 × 10^13

pruned = PruneCandidates(all_candidates)
// Rule 1 (Divisible Tile): tile size 必须整除 problem size — 从现有工作 [55]
// Rule 2 (Cluster Size): 每个 GEMM 的 cluster dim product ≤ 16 (H100 限制)
//                        且连续 GEMM 的 cluster dims 必须一致
// Rule 3 (Activation): accumulation dim 必须在 innermost loop
// Rule 4 (Dependency): L dim 不能设为 spatial
// Rule 5 (Memory Capacity): tensor 不能超过最低层 cache 容量
// 结果: ≈ 1.15 × 10^6 candidates

top_k_list = []
foreach (s, t, r) in pruned:
  (DV, plan) = DataflowAnalyzer(g, d, s, t, r)  // 见 Algorithm 1
  est_cost = CalculateCost(DV)  // C_l = V_l / B_l, minimax over layers
  top_k_list = UpdateTopKList(top_k_list, (est_cost, plan), k=11)

p_best = ProfileBestFromList(top_k_list, d)
// 将 top-11 candidates 编译为 CUDA kernel，H100 实测选最优
```

Cost Model: C_l(T_l) = V_l(T_l) / B_l，目标 minimax min_{T_1...T_L} max_{l=1...L} C_l(T_l)。约束 U_l(T_l) ≤ Cap_l。核心思想：找出所有内存层中数据传输最慢的那层（瓶颈），然后优化 tiling 策略使这层的搬移时间最小。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

1. **Pruning Rules 实现**：5 条规则在 Python 前端以 filter 函数实现，按级联顺序逐步过滤（Table III 显示各步 reduction rate）
2. **Cost Model**：分析性模型（非 ML-based），V_l 由 Dataflow Analyzer 计算，B_l 从硬件 spec 获取（H100: reg ~20TB/s, SMEM ~20TB/s, DSM variable, HBM 3TB/s）
3. **Top-K Selection**：K=11 经实验验证（Figure 12b），小于 11 时 accuracy 骤降，大于 11 时 improvement diminishing。Compilation time overhead 主要是 top-K kernel 的编译时间，cost model 预测仅 1-2s
4. **Search Engine vs Brute-Force**：对 G3/G4/G5，search engine 编译加速 12.25×/29.05×/68.26×（Table VIII）
5. **Offline compilation**：搜索和编译离线完成。Runtime 通过 binning/table lookup 按动态变化的 M 维度选择预编译 kernel（N, K, L 在模型编译时固定）

涉及论文标题：
- FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection

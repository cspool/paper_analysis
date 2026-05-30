## FlashFuser Fusion Search Engine (Cost Model + Pruning)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FlashFuser Search Engine 是 FlashFuser 编译框架的搜索组件，用于从由 DSM 引入的极大搜索空间（LoopSchedule 41种 × TilingSize cluster-level 5^4 × block-level 多种 × ResourceMapping）中高效选择最优 fused kernel 执行计划。核心设计：(1) Analytical Cost Model——minimax 公式 min max_l (V_l/B_l)，将 total cost 定义为所有 memory level 中 bottleneck stage 的 data movement time；(2) 5 条 Pruning Rules——结合 1 条来自 MC-Fuser 的规则 (Divisible Tile Sizes) 和 4 条新增的 DSM-specific 规则。

从编译框架角度拆解术语：

Cost Model：
$$C_l(\mathcal{T}_l) = \frac{V_l(\mathcal{T}_l)}{B_l}$$

Minimax optimization:
$$\min_{\mathcal{T}_1, \dots, \mathcal{T}_L} \left\{ \max_{l=1, \dots, L} \left( C_l(\mathcal{T}_l) \right) \right\}$$

subject to memory capacity constraints:
$$U_l(\mathcal{T}_l) \le \operatorname{Cap}_l, \quad \forall l \in \{1, \dots, L\}$$

5 Pruning Rules 的执行顺序和效果（GPT-6.7B）：
1. Divisible Tile Sizes (from MC-Fuser): 2.75×10^13 → 1.14×10^8 (>99.99% reduction)
2. Cluster Size Constraint: product≤16 hardware limit, consecutive GEMMs' cluster dims must match → 2.47×10^7 (78.3% red.)
3. Activation Constraint: accumulation dim must be innermost loop → 1.44×10^7 (41.5% red.)
4. Dependency Constraint: L dim can't be spatial (intermediate C can't communicate across spatial L) → 9.62×10^6 (33.3% red.)
5. Memory Capacity Limit: tensor ≤ lowest-level cache capacity → 1.15×10^6 (88.0% red.)

Search Algorithm (Algorithm 2): EnumerateAllCandidates → PruneCandidates → foreach candidate: DataflowAnalyzer → cost model → UpdateTopKList (K=11) → ProfileBestFromList on H100. K=11 chosen because accuracy approaches 100% of true optimum.

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FlashFuser 前端用 Python 实现搜索引擎。离线执行——对每个 model 的 GEMM chain configuration 预先搜索编译，生成 kernel library。运行时仅 M 维度动态变化（N, K, L 固定），通过 binning + table lookup 选择预编译 kernel。编译加速 vs brute-force: 12-68× (Table VIII, 362s-381s vs 1.2-8.1 hr)。搜索成本主要由 cost model prediction (1-2s/candidate) 和 Top-K 编译 profiling 组成。To generalize to other architectures (e.g. Cerebras WSE with mesh interconnect), the search engine's cluster-aware pruning rules would need adaptation to the specific topology constraints.

涉及论文标题：
- FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection

## Kuhn-Munkres Algorithm（KM 算法 / Hungarian Algorithm）for Bipartite Matching

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Kuhn-Munkres (KM) 算法，又称 Hungarian Algorithm（匈牙利算法），是求解加权二分图最小权（或最大权）完美匹配的经典组合优化算法。原始版本由 Kuhn (1955) 和 Munkres (1957) 提出，时间复杂度 O(n⁴)。改进版本（Wong 1979）通过维护最小 slack 值数组（minSlack），将每次增广的 slack 更新从 O(n²) 降至 O(n)，使总复杂度降为 O(n³)。KM 算法是 assignment problem（指派问题）的标准解法：给定 n 个工人和 n 个任务，每个工人-任务对有已知成本，求成本最小的完美分配方案。NetMoE 将 sample placement 问题转化为两个嵌套的 assignment problem，使用 KM 算法在多项式时间 O(I³) 内求解。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 NetMoE 中，KM 算法作为离线求解器（offline solver）被集成到训练框架中，而非编译框架。但从优化方法角度看，KM 算法属于整数线性规划（ILP）的多项式时间特殊解法：

**问题转化流程**：
1. **原始 ILP 问题**（Eq. 5-7）：`argmin max(t_intra, t_inter)`，受约束每个 device 容纳 I/J 个 samples。
2. **两阶段拆分**：Stage 1 优化 inter-node 通信（Eq. 6），Stage 2 优化 intra-node 通信（Eq. 7）。
3. **(0,1)-ILP 转化**：引入 `p_{i,n} ∈ {0,1}` 表示 sample i 是否在 node n，`p'_{i,j} ∈ {0,1}` 表示 sample i 是否在 device j。约束变为 `Σ_i p_{i,n} = I/N` 和 `Σ_n p_{i,n} = 1`。
4. **二分图构建**：左侧 P 为 I 个 samples，右侧 Q 为 N 个 nodes（每 node 复制 B = I/N 次，使 |Q| = I）。边权重 `W_{i,n} = c_{i,n}^{(l,gather)} + c_{i,n}^{(l+1,scatter)}`。
5. **KM 算法求解**：在 O(I³) 时间内找到最小权完美匹配，等价于求解 (0,1)-ILP 的最优解。

**KM 算法核心步骤（改进 O(n³) 版）**：
```
初始化：feasible labeling l(x) for x ∈ P∪Q, 匹配 M = ∅
while |M| < n:
    构建相等子图 E_l = {(x,y) | l(x)+l(y)=W(x,y)}
    从未匹配节点开始 BFS/DFS 搜索增广路
    if 找到增广路 P:
        M = M ⊕ P  (augment)
    else:
        计算 δ = min(l(x)+l(y)-W(x,y)) over reachable x and unreachable y
        更新 labeling: l(x) -= δ for reachable x∈P, l(y) += δ for reachable y∈Q
        （使用 minSlack 数组维护 δ，避免 O(n²) 重新计算）
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- Python 实现：
  - **SciPy**：`scipy.optimize.linear_sum_assignment(cost_matrix)`（O(n³)，基于修改的 Jonker-Volgenant 算法）
  - **PuLP**：通用 ILP 求解器，可解 assignment problem 但速度远慢于 KM（NetMoE Table 4: I/J=4 时 PuLP 50.1ms vs KM 0.48ms）
  - **scikit-learn**：`sklearn.utils.linear_assignment_` (deprecated in favor of scipy)
- C++ 实现：常用于竞赛编程的 O(n³) 模板（基于 minSlack 优化）
- NetMoE 的 KM 实现：在 C++/CUDA 中实现，运行于 CPU（后台线程），与 GPU 计算/通信重叠
- NetMoE Table 4 显示 KM 求解时间随 I/J 增长：I/J=2→0.08ms, 4→0.48ms, 8→1.48ms, 16→10.82ms, 24→31.09ms，均小于对应的 scatter+computation 时间
- 其他分布式系统中的应用：SpotServe（ASPLOS 2024）使用 KM 算法求解 context migration 的最优 device mapping

涉及论文标题：
- NetMoE: Accelerating MoE Training through Dynamic Sample Placement

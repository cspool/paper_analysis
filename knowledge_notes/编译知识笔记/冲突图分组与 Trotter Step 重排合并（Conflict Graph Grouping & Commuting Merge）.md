## 冲突图分组与 Trotter Step 重排合并（Conflict Graph Grouping & Commuting Merge）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 这是 Kernpiler 编译流水线第二阶段（4.3 节，Algorithm 2）：对每个 Trotter step 的部分 Trotter 化 unitary 集合，构造 conflict graph（顶点=unitary，两个 unitary 不对易 [t_i,t_j]≠0 时连边），用贪心最大独立集（greedy max independent set）把互相对易的 unitary 分组成"可交换 group"（commuting group）。随后：(1) 排序——把最大的两个 group 放在 Trotter step 的两侧（edges），且相邻 Trotter step 中这两个 group 互换位置，使相邻 step 的边缘 group 相同；(2) 合并——由于对易性，相邻 step 相同的 group 可跨 step 重排合并：e^{iH_it}e^{iH_it}→e^{i2H_it}，把 unitary 数减半且零额外误差；(3) 随机化——group 内项顺序每 step 随机 shuffle（见 Randomized Compilation 条目），把 coherent 误差转 stochastic。目标：在保持误差不受损的前提下最大化门取消（gate cancellation）与 term 合并。
- 与 Paulihedral 等"单 step 内重排"的区别：已有工作（如 travelling-salesman 式 Trotter step 重排，Schmitz 2023）把重排限制在单个 step 内；本论文的分组目标是跨 Trotter step 合并，且同时沿"深度"与"门数"两个轴优化（避免传统的 depth-gatecount tradeoff）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 计算流程（图 3 + Algorithm 2）：
```
BuildConflictGraph(H): 节点=term/unitary；[t_i,t_j]!=0 时加边
GreedyCommutingGroups(G): while G 非空: I=GreedyMaxIndependentSet(G); 追加; 删除 I
ReorderTrotterSteps({H_1..H_n}):
  每 step: 建冲突图 -> 贪心分组 -> 组内随机化顺序 -> 对易 group 连续拼接
  重排相邻 step；对 [A,B]=0 (A in step k, B in step k+1) 合并
  输出: 修改后的 Trotter steps
```
- 例子（图 3，1D Ising）：输入部分 Trotter 化 unitary 集合 → Step 1 冲突图 + 两个独立集 G1/G2（互交换 group）→ Step 2 G1/G2 放两侧、相邻 step 翻转 → 相同 group 相邻可合并（e^{iH_it}e^{iH_it}→e^{i2H_it}）→ Step 3 组内 shuffle。深度优化的收益来自：不对重排做深度-门数折中（论文把 MCTS 用于门数、重排用于深度，两轴并行优化）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：图算法（冲突图构建、贪心最大独立集）在 CPU（AMD EPYC 9654P）上运行；reorder pass 运行时占比较小（图 7：大多数 benchmark <0.1s，仅 PD1-super 222 ~159s），远小于 MCTS rewrite pass（主导，58-5397s）。使用时作为流水线第二 pass 紧接 partitioning；对 bipartite 冲突图（两分图）的哈密顿量，两个 commutator group 覆盖大部分 step，翻转排序可产生接近 Δt³ 的误差缩放（一阶 Trotter 大幅受益）。

涉及论文标题：
- Kernpiler: Compiler Optimization for Quantum Hamiltonian Simulation with Partial Trotterization

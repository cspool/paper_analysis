## Variable Interaction Graph（VIG，变量交互图）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Variable Interaction Graph（VIG）是 SAT 实例的结构化图表示：节点 = 布尔变量，边 = 两变量在至少一个子句中共现。VIG 是 SAT 结构分析的常用工具（揭示变量间相关性/约束耦合）。SATIC 使用加权 VIG 作为编译的中间表示：边权重 = 两变量在子句中的共现次数（图 4 示例：(x1∨¬x2)∧(¬x1∨x2)∧(x1∨¬x2∨x3) 中 x1-x2 共现 3 次、x1-x3 与 x2-x3 各 1 次）。VIG 的价值：变量邻接即约束耦合，邻域变量强相关，BFS 沿 VIG 选变量可保证 clause-completeness；且 VIG 规模（|V| 个节点）远小于公式化后 QUBO（含 ancillary），大幅降低图遍历成本。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 SATIC 编译框架中，VIG 是变量选择/子问题形成的核心 IR（Algorithm 1：CreateGraph(CNF) → VIG）。运转流程：
```
1. VIG ← CreateGraph(CNF)             # 节点=变量，边权=共现次数
2. # SATIC++：Limited Neighbors 剪枝
   MST ← MaxSpanningTree(VIG)          # 保留全连通（高权重边优先，O(E log V)）
   for node: 恢复被删边（按权重降序）至每节点邻居上限 N=10（≈半容量）
3. for iteration:
     root ← randint(1, n)
     var_list ← BFS(VIG, root)         # 按距 root 距离排序（Neighbor Shuffling 时每层随机置换邻接表）
     Q_sub ← UnitProp(CNF, var_list, S_global)   # freeze+传播+公式化
     S_sub ← IsingHardware(Q_sub); 更新 S_global; CheckSolution
```
对比：50 变量/200 子句 3SAT（transition region）公式化为 Chancellor's 后是 250 变量 QUBO，若在 QUBO 层选变量需面对 250 个节点；SATIC 只在 50 节点 VIG 上 BFS，k 越大优势越明显（4SAT 比值 ≈10）。另提供 Web Graph 可视化（径向布局，中心 = 高连接度变量）辅助诊断稠密/稀疏区域与二分结构。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：从 CNF 单遍扫描构建（哈希表统计变量对共现次数），稠密 kSAT 实例中节点度数可高达 80（Batch-4-100-1000）；加权边支持 MST 与按权重剪枝。使用：SATIC 用它做子问题变量选择（BFS 邻域）、clause-completeness 保证、可视化分析（Web Graph）；VIG 也是其他 SAT 结构分析/社区发现工作（图聚类、community detection）的基础。

涉及论文标题：
- SATIC: An Optimizing Ising Compiler for SAT(isfiability)

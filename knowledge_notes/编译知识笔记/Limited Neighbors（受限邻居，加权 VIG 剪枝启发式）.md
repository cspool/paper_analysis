## Limited Neighbors（受限邻居，加权 VIG 剪枝启发式）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Limited Neighbors 是 SATIC 的第一个启发式技巧（Intermediate Representation 类）：剪枝加权 VIG，使每个变量只保留连接最强的 top-N 邻居（N 按目标 Ising 机器容量预先确定，45-spin 芯片取 N=10≈半容量）。两步骤：① 构建加权 VIG 的 Maximum Spanning Tree（MST，优先高权重边、保证全图连通，丢弃其余边，O(E log V)）；② 细化：逐节点按权重降序逐步恢复被删边，直到每节点邻居数达到上限 N。作用：稠密 kSAT（尤其高 k）中变量度数可远超子问题规模限制（Batch-4-100-1000 度数达 80，而子问题仅约 20 变量），Limited Neighbors 提炼最相关连接、压缩 VIG，加速迭代 BFS 遍历，并让 BFS 探索更深的"结构相关区域"。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 SATIC++ 编译框架中，Limited Neighbors 是子问题形成前的图预处理 pass：
```
# 输入：加权 VIG（节点=V，边权=共现次数）
# Step 1: MST ← MaximumSpanningTree(VIG)      # 高权重边优先，保连通，O(E log V)
# Step 2: for node in V:                        # 细化恢复
#     for edge in sorted(被删边[node], by weight desc):
#         if degree(node) < N: restore(edge)    # N=10（45-spin 芯片）
# 输出：每节点 ≤N 邻居的稀疏 VIG
# 后续：BFS(VIG', root) → 更深度、更相关的邻域 → 子问题质量更高
```
消融数据（Fig.14）：Limited Neighbors 影响最大（B0 vs B2、A4 vs A5、C0 vs C1）；Batch-4-100-1000 上把 solved 从 85（Neighbor Shuffling 后）提到 88；代价是部分实例 successful repeats 略降（1,347→1,321，局部冗余减少）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Prim/Kruskal 类 MST 算法 + 按权重排序的边恢复；每实例一次性预处理（编译开始），复杂度 O(E log V)，仅 N 是硬件相关旋钮。使用：在高 k/稠密 SAT（度数爆炸）场景最有效；与 Neighbor Shuffling 协同（剪枝后的受控随机化收益最大）；Web Graph 可视化可观察剪枝前后结构（Fig.5）。

涉及论文标题：
- SATIC: An Optimizing Ising Compiler for SAT(isfiability)

## Local Search（局部搜索重启）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Local Search 是 SATIC 的运行时优化启发式：经过预设迭代次数（T_LS）无解后，随机重排全局解向量 S_global 中的值，从而逃离局部极小——比硬重启（hard restart，重跑全部预处理）更轻量，且可按具体问题实例定制 T_LS。在随机求解器中，局部搜索/重启是常见收敛机制（类似 WalkSAT 的随机翻转）；SATIC 用它在不重跑编译预处理的前提下给搜索新的起点。评估：T_LS 与典型求解时间大致匹配时最有效（Batch-4-50-500 上帮助大），设置过激进则有害（UF75 上过早重启打断接近收敛的运行，B0 vs B1 变差）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 SATIC++ 编译框架的迭代主循环中：
```
iteration ← 0; S_global ← RandomAssignment()
while iteration < max_iter:
    root ← randint(1, n); var_list ← BFS(VIG, root)
    Q_sub ← UnitProp(...); S_sub ← IsingHardware(Q_sub)
    S_global ← merge(S_global, S_sub)
    if CheckSolution(CNF, S_global): return SAT
    if iteration % T_LS == 0:             # Local Search 触发（T_LS 由 profile 设定）
        S_global ← random.shuffle(S_global)   # 轻量重启，不重跑预处理
    iteration += 1
```
复杂度 O(n)（一次洗牌）；T_LS 是 SATIC++ 的两个超参数之一（另一为 Limited Neighbors 的 N）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：迭代计数 + 模 T_LS 触发随机洗牌全局解向量；T_LS 每次 workload 用小型 profiling 运行标定。使用：配合其他 tricks 提升难实例的最终收敛（评估中从 10K 迭代提到 50K 迭代 + 各 trick 后 repeats 显著上升）；作为无重编译开销的收敛兜底机制。

涉及论文标题：
- SATIC: An Optimizing Ising Compiler for SAT(isfiability)

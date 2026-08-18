## Subproblem Formation（子问题形成）与变量冻结（Variable Freezing）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Subproblem formation（子问题形成，即问题分解 decomposition）是把超出 Ising 机器容量（spin 数）的大问题切成能装进硬件的小子问题（subproblem/sub-QUBO）的过程。经典 SAT 求解器也用分解（并行/简化），但 Ising 导向的分解器专门为满足 spin 数与连接度约束而重构问题。SATIC 的核心创新是"先分解后公式化"：在 CNF 层（尚无 ancillary）做变量选择与子问题形成，而不是像 qbsolv/D-Wave EID 那样先在 QUBO 层公式化再分解。变量冻结（freezing）：从 BFS 变量表 L 底部逐个取未选变量，赋予全局解向量 S_global 中预定值，之后该变量不再出现在子问题中（其约束通过单元传播影响剩余子句）。全局解向量 S_global 是运行中的部分解容器，子问题解出后回填更新，CheckSolution 验证全局满足后终止。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 SATIC 编译框架中（Algorithm 1 / Fig.3 流程）：
```
for iteration < max_iter:
    root ← randint(1, n);  var_list ← BFS(VIG, root)
    Q_sub ← UnitProp(CNF, max_var, var_list, S_global):   # 子问题形成核心
        Q_sub ← Formulate(CNF); Q_sub.size ← CheckSize(Q_sub)
        while Q_sub.size > machine.capacity:              # 45 spins
            CNF_sub ← VarFreeze(CNF, var_list, S_global)  # 冻结 L 底部变量
            Q_sub ← Formulate(CNF_sub); Q_sub.size ← CheckSize(Q_sub)
        return Q_sub
    S_sub ← IsingHardware(Q_sub)                          # 硬件解子问题
    S_global ← S_sub (合并)；SAT ← CheckSolution(CNF, S_global)
```
示例（4-spin 机器，Eq.(5)）：选 {x1,x2,x3}，冻结 x4=0、x5=0 → 子 CNF (x1∨x2∨x3)∧(¬x3) → 公式化 4 变量恰好装满。若分解发生在 QUBO 层，7 变量 QUBO 选 4 变量有 C(7,4)=35 种选法，多数破坏 clause 语义（见 Ancillary-awareness/Clause-completeness 条目）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：SATIC 用加权 VIG + BFS + freeze + 单元传播实现（O(TL) 总复杂度，T 迭代数、L 文字数）；QUBO 层分解器（qbsolv/EID）用能量影响启发式切 QUBO。使用：配合 Limited Neighbors（剪枝邻域）、Neighbor Shuffling（BFS 随机化）、Bulk Freeze（批量冻结）与 Ancillary Estimation（快速规模检查）；Subproblem 大小由 machine.capacity 决定（45-spin 芯片每次迭代约 20 变量子问题）。

涉及论文标题：
- SATIC: An Optimizing Ising Compiler for SAT(isfiability)

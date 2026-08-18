## QUBO（Quadratic Unconstrained Binary Optimization，二次无约束二值优化）与 Ising 哈密顿量

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
QUBO 是对二值变量 $x_i \in \{0,1\}$ 的二次无约束优化形式：$H(x) = x^{\top} Q x$，其中 Q 是实值 $n \times n$ 矩阵（可整理为对角项 + 二次项，对应线性与双线性系数）。QUBO 与 Ising 模型同构：通过 $s_i = 2x_i - 1$ 双向转换（$s \in \{-1,+1\}$，Ising 能量 $H(s) = -\sum_{i<j}J_{ij}s_i s_j - \sum_i h_i s_i$），二者可无缝互转，因此同一编译器输出可同时喂给 Ising 硬件与 QUBO 软件求解器（如 D-Wave Tabu）。在 SAT 场景，SATIC 在子问题形成后把（子）CNF 公式化为 QUBO/Ising 参数，每个子句贡献固定惩罚（不满足时）+奖励（满足时）。QUBO 公式化的开销主要是引入 ancillary（辅助）变量，使 QUBO 规模大于原 SAT 变量数。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 SATIC 编译框架中，QUBO 是"数学公式化"阶段的目标表示（编译流水线：CNF → 子问题形成 → QUBO 公式化 → Machine Embedding → Ising 硬件）。具体流程（Chancellor's 公式化 3SAT 子句 $C_1=(x_1\vee x_2\vee x_3)$，Eq.(3)）：
```
H_C1(x1,x2,x3) = min_{a1∈{0,1}} [ -x1 -x2 -x3 + x1x2 + x1x3 + x2x3 - a1(x1+x2+x3-2) ]
# 拆成 QUBO 系数（Q 矩阵）：
#   线性项: x1,x2,x3 各 -1；a1 隐含
#   二次项: x1x2, x1x3, x2x3 各 +1；-a1x1, -a1x2, -a1x3 各 -1；+2a1
```
（子）QUBO 规模检查（CheckSize）决定子问题是否适配 Ising 机器 spin 容量；规模用 Ancillary Estimation 按子句宽度 k 查表估算（Chancellor's 约 (2k-5) ancillary/kSAT 子句）避免重复公式化。Batch-4-100-1000 的等价 QUBO ≈3100 变量（Chancellor's）/2100（ILP），相对 45-spin 容量比值 69×–47×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：编译器/求解器库（D-Wave Ocean SDK 的 BinaryQuadraticModel、dwave-tabu、qbsolv）接受 QUBO（Q 矩阵）或 Ising（J,h）表示；SATIC 内部 Formulate 例程把子 CNF 转 QUBO（含 Chancellor's、ILP、Flat ILP 三套公式化，按子句 k 混合）。使用：QUBO 矩阵元素受 Ising 硬件系数范围约束（SATIC 测试床 [−14,+14]），超出需缩放（Dynamic Upscaling）或合并 spin（Adaptive Spin Merging）；D-Wave Tabu 作为软件后端接受同一 45 变量 QUBO 子问题（无限系数范围、20ms/次）。QUBO 还可编码 Max-Cut、图着色、背包等众多 COP。

涉及论文标题：
- SATIC: An Optimizing Ising Compiler for SAT(isfiability)

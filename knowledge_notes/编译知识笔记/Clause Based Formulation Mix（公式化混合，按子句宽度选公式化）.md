## Clause Based Formulation Mix（公式化混合，按子句宽度选公式化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Clause Based Formulation Mix 是 SATIC 的公式化类启发式：按子句的宽度 k 为每个子句选用最省 ancillary 的 SAT-to-QUBO 公式化。CNF 层子问题形成 + 单元传播后，子 CNF 通常含混合宽度的子句（高 k 与低 k 并存）。策略：k≤3 用 Chancellor's（对低 k 更省 ancillary），k>3 用 ILP/Flat ILP（对高 k 更省），从而显著降低子问题 ancillary 总数、让有限容量 Ising 机器装入更多问题变量。两种公式化的能量语义（Chancellor's 奖励 −1/惩罚 0；ILP 奖励 0/惩罚 +1）虽数值范围不同，但谱能量距离相同、保持问题结构，因此子句级混合数学安全。它与 Negative Literal Inversion 协同最好。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 SATIC++ 编译框架的 Formulate pass 中（每子句独立决策）：
```
for clause c in CNF_sub:
    k ← width(c)                      # 单元传播后宽度已降低
    if k <= 3:  use Chancellor's(c)   # 2SAT/1SAT/3SAT：ancillary 少
    elif flat:  use FlatILP(c)        # k>3：ancillary log2(k)/子句，系数摊平
    else:       use ILP(c)            # k>3 标准 ILP（ancillary 最少）
    # 特殊：2SAT 子句（传播后常见）ILP/Flat ILP 表现差 → Chancellor's
Q_sub ← sum(energies)；系数按硬件范围后处理（Upscaling/Spin Merging）
```
评估（Batch-4-100-1000）：单元传播后 2SAT 子句频出，ILP/Flat ILP 在 2SAT 上表现差；Flat ILP（高宽）+ Chancellor's（2SAT）组合把 solved 从 94 提到 97、successful repeats 从 3,552 提到 4,676；最终配置改用标准 ILP + Chancellor's + Adaptive Spin Merging 把 repeats 提到 6,816（最优）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：编译框架内按子句宽度分发的公式化分派表（查表选择 Chancellor's/ILP/Flat ILP）+ 子问题级系数合并；复杂度 O(m_s)（子问题子句数）。使用：最大化硬件容量利用率的关键公式化技巧；需与系数处理（Dynamic Upscaling、Adaptive Spin Merging）配合应对 ILP 的宽系数范围。

涉及论文标题：
- SATIC: An Optimizing Ising Compiler for SAT(isfiability)

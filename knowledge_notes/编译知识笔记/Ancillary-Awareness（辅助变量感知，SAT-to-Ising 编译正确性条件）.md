## Ancillary-Awareness（辅助变量感知，SAT-to-Ising 编译正确性条件）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Ancillary-awareness 是 SATIC 论文定义的 SAT-to-Ising 编译正确性条件之一：形成子问题时，每个子句（clause）的 ancillary 变量必须与其对应的问题变量同组（一起进入同一个子问题）。理由：QUBO/Ising 公式化在每个子句粒度引入能量奖励/惩罚语义，若子问题只含问题变量而排除该子句的 ancillary，求解器只能把被排除的 ancillary 定成固定值，从而无法保证该子句的能量景观与原始目标函数对齐（能量景观错位）。论文用 Table I 量化：3SAT 子句 C1 的 ancillary a1 被固定为 0 时，满足赋值 i=7（x1=x2=x3=1）与唯一不满足赋值 i=0 能量同为 0，无法区分。在 transition region 解稀疏的硬实例中，这种错位显著降低求解概率。D-Wave 的 qbsolv/EID 等状态最优分解器忽略此条件（按能量影响选变量时，低度的 ancillary 最易被排除）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 SATIC 编译框架中，ancillary-awareness 靠"在 CNF 层（ancillary 尚不存在）选变量"按构造成立。运转流程：
```
# 目标：子问题 (x1,x2,x3) 必须带上 C1 的 ancillary a1 → 完整保留 C1 能量语义
# SATIC（CNF 层分解）：选变量 x1,x2,x3 → freeze x4,x5 → 公式化自动引入 a1 → 子问题含 (x1,x2,x3,a1)
# 反例（QUBO 层分解，D-Wave EID）：QUBO 层 7 变量选 4 → 选 {x1,x2,x3,x4} 排除 a1,a2
#   → a1,a2 被固定值 → C1,C2 能量语义失真（违反 ancillary-awareness）
# 受控实验（UF75，50K 迭代，100 实例）：
#   违反 ancillary-awareness（但 clause-complete）→ 仅 44/100 解出
#   D-Wave EID++（每次迭代随机化 ancillary 值）仅部分缓解（诊断性启发式，非按构造保证）
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：编译框架层面把变量选择约束到"按子句整体"进行（选择变量时自动包含其所在子句的全部文字及其公式化 ancillary），或在 CNF 层分解（SATIC 做法）。使用：任何 SAT-to-Ising/QUBO 编译器都应检查此性质；D-Wave EID++ 是论文构造的对照变体（随机化 ancillary 值，性能介于 EID 与 SATIC 之间），证明 ancillary 处理对收敛的因果重要性。相关条目：Ancillary Variable、Clause-completeness。

涉及论文标题：
- SATIC: An Optimizing Ising Compiler for SAT(isfiability)

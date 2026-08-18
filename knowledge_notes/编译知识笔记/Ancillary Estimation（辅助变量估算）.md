## Ancillary Estimation（辅助变量估算）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Ancillary Estimation 是 SATIC 的运行时优化启发式：通过统计（子）CNF 的子句宽度（k）分布，用查表法直接估算子问题公式化为 QUBO 后的规模（ancillary 数），避免为"检查子问题是否适配硬件容量"而做完整公式化。对给定公式化，按 k 预计算每 kSAT 子句的 ancillary 开销查表（如 Chancellor's 约 (2k−5) ancillary/子句；ILP 约 log₂(k)），运行时对子 CNF 线性扫描求和。每个 size check 从"完整翻译到 QUBO"降为"线性扫描 + 查表"，论文评估 size check 时间减少近 9×。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 SATIC 编译框架的 UnitProp/子问题形成循环中：
```
# 原本：CheckSize(Q_sub) 需 Formulate 后数变量
# 改进：用 Ancillary Estimation 预估算
def estimate_QUBO_size(CNF_sub):
    n_anc = 0
    for c in CNF_sub:
        k ← width(c)
        n_anc += LOOKUP[formulation][k]   # 如 Chancellor's: 2k-5; ILP: log2(k)
    return n_vars(CNF_sub) + n_anc
while estimate_QUBO_size(CNF_sub) > machine.capacity:   # 线性扫描/次
    CNF_sub ← VarFreeze(...)   # 冻结变量 → 子句变窄 → 估算变小
```
配合 Bulk Freeze（批量冻结 + 一次传播）进一步减少重复检查开销。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：编译框架内建 ancillary 开销查表（按公式化 × 子句宽度 k），size check 用线性扫描（O(L_s)）而非完整公式化。使用：容量预测可靠（硬件容量已知）时的标准优化；与 Bulk Freeze、Clause Based Formulation Mix 等一起构成 SATIC++ 的运行时优化包（Runtime Optimization Tricks）。

涉及论文标题：
- SATIC: An Optimizing Ising Compiler for SAT(isfiability)

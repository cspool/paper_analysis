## qbsolv / D-Wave EID（Energy Impact Decomposer，QUBO 层分解器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
qbsolv 是 D-Wave 开源的 QUBO 分解器（decomposer）：把大型 QUBO 切分为可装入量子/经典退火器的小 sub-QUBO，用能量影响（energy impact）启发式迭代求解并合并结果。D-Wave EID（Energy Impact Decomposer）是 qbsolv 的最新演进版本，属于文献中最好的通用分解器之一；它按变量对能量的影响选变量形成子问题，考虑问题稀疏性与目标硬件连接度。两者都是"QUBO 层分解"的代表——先对问题做数学公式化（引入 ancillary），再在 QUBO 图上做子问题形成。SATIC 论文以 D-Wave EID 为主要 baseline（取自开源 dwave-hybrid 包，按官方文档方法使用），并构造了修改版 D-Wave EID++（每次迭代随机化 clause 特定 ancillary 值，作为诊断性启发式，证明 ancillary 处理的重要性）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
D-Wave EID 的编译/求解流程（QUBO 层分解范式）：
```
# 输入：完整 QUBO（如 Batch-4-100-1000 → Chancellor's ≈3100 变量）
while not converged:
    for v in variables:  impact[v] ← 能量影响估计（剔除 v 对全局能量的改变）
    sub ← top-k 高 impact 变量（+ 相关变量，考虑硬件连接度）
    S_sub ← Ising/Tabu 求解 sub-QUBO（fixed 其余变量）
    更新全局解；能量改进则继续，否则终止
```
痛点（论文对比）：① 每迭代在完整 QUBO 上重复分解 → 软件运行时随问题规模爆炸（UF20→UF250 规模增 12.5× 时 EID 每迭代 runtime 增 10.6×，SATIC++ 仅 1.6×；端到端 EID 18,877ms vs SATIC++ 69.3ms，272.4×）；② ancillary 是低能量影响节点 → 最易被排除 → 违反 ancillary-awareness（EID 在 UF20 以上即失败）；③ D-Wave EID++（随机化 ancillary 值）性能介于 EID 与 SATIC 之间，但非按构造正确。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：开源 dwave-hybrid Python 包（https://pypi.org/project/dwave-hybrid/0.6.12/）中的 Energy Impact Decomposer，按官方文档（docs.dwavequantum.com 的 Decomposition 章节）使用；子问题求解可接量子退火器或 Tabu。使用：作为通用 QUBO 求解 baseline 与 SATIC 对比（平均迭代数、TTS、runtime）；结论是 QUBO 层分解因 ancillary 失真与重复公式化开销显著劣于 CNF 层分解的 SATIC。

涉及论文标题：
- SATIC: An Optimizing Ising Compiler for SAT(isfiability)

## D-Wave Tabu solver（dwave-tabu，经典 QUBO/Ising 软件求解器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
D-Wave Tabu 是 D-Wave 开源的经典（非量子）QUBO/Ising 求解器（GitHub: dwavesystems/dwave-tabu，论文引用版本 0.5.0），基于禁忌搜索（tabu search）局部搜索算法，对中小规模 QUBO 高效。在 SATIC 论文中，Tabu 被用作"替换 Ising 芯片的软件后端"来证明 SATIC++ 的硬件无关性：Tabu 接受 SATIC++ 产生的同一批 45 变量 QUBO 子问题。对比：Ising 芯片系数范围有限（[−14,+14]）但快（200μs 最大退火时间）且能效高（4.8μJ/迭代）；Tabu 无限系数范围但 20ms 最大退火时间（timeout）、约 3.3J/迭代——Tabu 约慢 2 个数量级、能耗高约 5 个数量级。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
作为求解后端的运转流程（硬件无关性实验）：
```
# SATIC++ 编译输出不变：CNF → VIG → 子问题形成 → 45 变量 QUBO
for iteration:
    Q_sub ← SATIC++ 子问题（45 变量 QUBO）
    if backend == Ising_chip:  S_sub ← chip 退火(Q_sub)     # 200μs, 4.8μJ, 系数 [-14,+14]
    elif backend == Tabu:      S_sub ← Tabu.sample(Q_sub, timeout=20ms)   # 无限系数, 3.3J
    S_global ← merge(S_global, S_sub); CheckSolution(CNF)
# 结论：SATIC++ 在两个后端上都解出全部实例（如 UF175 全解、UF250 92/100）
#       → 证明编译逻辑与硬件解算器解耦（硬件无关的全局 Ising/QUBO 编译器）
```
Tabu 在评估中还作为 D-Wave EID 子问题的求解器（EID 用能量启发式切 QUBO 后交给 Tabu/退火器），因此 EID 的失败主要归因于分解层（ancillary-unaware）而非 Tabu 本身。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：C++ 禁忌搜索采样器（Python 绑定），输入 QUBO/Ising（BQM），输出多样本解；开源于 GitHub（dwave-tabu）。使用：D-Wave 混合求解（qbsolv/EID 的子问题求解器）、SATIC++ 的软件后端对比、无限系数范围场景；性能特征：无系数范围限制但慢且耗能（20ms/3.3J vs Ising 芯片 200μs/4.8μJ），约 2 个数量级慢、5 个数量级能耗高。

涉及论文标题：
- SATIC: An Optimizing Ising Compiler for SAT(isfiability)

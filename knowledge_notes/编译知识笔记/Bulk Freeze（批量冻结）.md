## Bulk Freeze（批量冻结）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Bulk Freeze 是 SATIC 的运行时优化启发式，优化默认流程中逐变量冻结 + 每次重跑单元传播的高开销。默认流程：每次只冻结一个变量、重跑单元传播、检查子问题是否适配，直到适配。Bulk Freeze：从初始迭代学习一个典型冻结批量大小 B，后续迭代投机地一次冻结 B 个变量再传播；若批量冻结后子问题仍超容量或明显欠利用，则回退到围绕 B 的逐一定制（fallback 保持适应性）。效果：显著减少重复单元传播 pass（论文评估"substantially decreases unit propagation time, even when accounting for its own overhead"）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 SATIC++ 编译框架的 UnitProp 例程中（Fig.6 Bulk Freeze Flow）：
```
# 学习阶段（首个迭代）：逐变量冻结，记录达到容量所需冻结数 B
# 稳态（后续迭代）：
while True:
    CNF_sub ← VarFreeze_batch(CNF, var_list, S_global, B)   # 一次冻结 B 个
    Q_sub ← Formulate(CNF_sub); size ← estimate/CheckSize
    if size > capacity:   # 过冻结 → 回退：解除 1 个再试（围绕 B 逐一定制）
        B ← B - 1; continue
    elif size << capacity:  # 欠利用 → 少冻结或接受（适配性保留）
        B ← B + 1   # 自适应
    else: break
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：运行时自适应批量（学习 B、投机冻结、越界回退）；复杂度 O(L_s) 级（单次批量传播）。使用：迭代次数多、子问题形成频繁的场景收益最大；与 Ancillary Estimation（快速规模检查）配合减少每轮的公式化/检查成本。

涉及论文标题：
- SATIC: An Optimizing Ising Compiler for SAT(isfiability)

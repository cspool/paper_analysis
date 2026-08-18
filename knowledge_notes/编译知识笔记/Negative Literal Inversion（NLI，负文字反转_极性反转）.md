## Negative Literal Inversion（NLI，负文字反转/极性反转）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Negative Literal Inversion（NLI）是 SATIC 的公式化类启发式：子问题形成后、QUBO 公式化前，若某变量在子问题中主要以负文字出现，则反转其极性使其在公式化中成为正文字（公式化后用 renaming 语义还原解）。动机：标准公式化（如 Chancellor's）对负文字用 ¬x→(1−x) 变换，导致"正文字多的子句"与"负文字多的子句"的 QUBO 系数略有不同，这种系数不对称会扭曲能量景观、引入局部能量陷阱，使 Ising 机更难找到全局解。NLI 通过变量重命名（renaming）消除这种不对称：变量数/子句数不变，仅把负极性为主的变量翻成正极性再公式化，解出后把被反转的变量值还原到原 SAT 表示。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 SATIC++ 编译框架的 Formulate pass 中：
```
# 输入：子 CNF（自由变量集合）
for x in free_vars:
    pos[x] ← #x 以正文字出现的次数;  neg[x] ← #x 以负文字出现的次数
    if neg[x] > pos[x]:                       # 以负为主 → 反转（NLI）
        rename x → x'（所有 ¬x 变成 x'，所有 x 变成 ¬x'）   # 记入 renaming 表
# 公式化：对极性已均衡的子句做 Chancellor's/ILP → 能量景观更平滑
# 硬件求解后：按 renaming 表还原 x' 的取值回原变量 x
```
NLI-1 按全局极性反转（适合 3SAT/UF75）；NLI-2 按子句宽度分别反转（适合 4SAT/Batch-4-50-500）。评估（Batch-4-100-1000）：NLI 把 solved 从 97 提到 100/100、successful repeats 从 4,676 提到 6,475；对 ILP 族公式化尤其有效（ILP 对负文字更敏感）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：编译框架内维护 renaming 映射（极性统计 + 替换 + 求解后还原）；复杂度 O(L_s)（子问题文字数）。使用：配合 Clause Based Formulation Mix 与 ILP 族公式化效果最好；是消除公式化引入的局部极小、提升 Ising 机成功率的低成本技巧。

涉及论文标题：
- SATIC: An Optimizing Ising Compiler for SAT(isfiability)

## ILP 公式化与 Flat ILP（kSAT-to-QUBO 整数线性规划式编码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ILP 公式化（论文引用的 [46] 中的 ILP-based 公式化）是面向 kSAT（k>2）的更紧凑 SAT-to-QUBO 编码：对 n 变量 m 子句 kSAT 用 n+m·log₂(k) 个 spin（ancillary 权重按 2 的幂增长），远少于 Chancellor's 的 n+m(2k−5)。其每个子句的能量语义：满足子句奖励 0、不满足子句惩罚 +1（与 Chancellor's 的 −1/0 奖励/惩罚范围不同，但谱能量距离相同，子句级混合数学安全）。代价：ILP 的 ancillary 权重按 2 的幂增长（4SAT 子句 2a2+1a1、7SAT 子句 4a3+2a2+1a1），系数范围宽，易超出 Ising 硬件系数范围而被截断。Flat ILP 是 SATIC 的改进：把高权 ancillary 拆成多个低权 ancillary（7SAT 的 4a3+2a2+1a1 → 2a4+2a3+2a2+1a1），在保持 ILP 奖励/惩罚语义的同时摊平系数范围，平衡 ancillary 数与系数范围。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 SATIC 编译框架中，ILP/Flat ILP 用于 k>3 子句的公式化（单元传播后 2SAT 子句则回退 Chancellor's，因 ILP 族在 2SAT 上表现差）。公式化与系数生成流程（k=4 子句，ancillaries a1,a2）：
```
# ILP:  ancillaries 权重 2a2 + 1a1（2 的幂），系数范围宽
# Flat ILP: 拆平为多个等/低权 ancillary，如 4a3+2a2+1a1 → 2a4+2a3+2a2+1a1
# 判断：若 max|coeff| 超出硬件系数范围([-14,+14]) 且子句宽 k>3 → 用 Flat ILP；
#        否则用标准 ILP（ancillary 更少）
```
评估中标准 ILP 与 Flat ILP 在 2SAT 上表现差（单元传播后 2SAT 子句频繁出现），所以 Clause Based Formulation Mix 用 Flat ILP（高宽子句）+ Chancellor's（2SAT 情形）组合，把 Batch-4-100-1000 的 solved 从 94 提到 97；后续用标准 ILP（非 Flat）+ Chancellor's + Adaptive Spin Merging 处理 Flat ILP 的系数问题，把 successful repeats 从 6,621 提到 6,816（最佳配置）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：编译器 Formulate 例程按子句 k 选择公式化（查表/决策逻辑），ILP ancillary 权重按 2^k 规律生成、Flat ILP 按位摊平重写系数；配合 Ancillary Estimation 的按 k 查表（ILP 开销 log₂(k)/子句）。使用：适合 k 大、ancillary 开销敏感的子问题；系数超范围时配合 Adaptive Spin Merging（动态合并未用 spin 扩系数范围）与 Dynamic Upscaling（缩放系数）使用。局限：ILP 对负文字更敏感（需配合 Negative Literal Inversion）。

涉及论文标题：
- SATIC: An Optimizing Ising Compiler for SAT(isfiability)

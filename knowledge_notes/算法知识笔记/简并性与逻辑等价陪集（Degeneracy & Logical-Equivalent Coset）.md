## 简并性与逻辑等价陪集（Degeneracy & Logical-Equivalent Coset）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
简并性（degeneracy）是量子稳定子码的关键属性：多个不同的物理错误模式产生完全相同的 syndrome 测量结果，且这些错误在物理上不可区分。任何错误 E 可唯一分解 E = s(E)·t(s)·l(E)：s(E)∈S 为稳定子分量，t(s) 为纯错误分量（仅由 syndrome s 决定，t(s)=∏_g T_g^{(1−s_g)/2}），l(E) 为逻辑分量。若 E1 = E2·S（S∈S），则 E1|ψ⟩ = E2S|ψ⟩ = E2|ψ⟩——两个错误对码字作用完全相同，称为退化错误。逻辑等价陪集（logical-equivalent coset）即固定 syndrome s 与逻辑错误 L 下所有 {E | E = S_g·t(s)·L, ∀S_g∈S} 构成的集合：同一陪集内的错误共享 syndrome 与逻辑效果，仅在稳定子变形上不同。核心结论：最可能的物理错误未必对应最可能的逻辑错误——最优解码应最大化逻辑后验 p(L|s) ∝ Σ_{E:l(E)=L} p(E) = Σ_{S∈S} p(E=St(s)L)，即对指数大小的稳定子群求和（coset ML），而传统 MWPM 求解的是物理 ML argmax p(E|s)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
本论文（ISCA 2026）的例子（Fig.2）：syndrome {1,2,3,4} 可匹配为 {1,4}∪{2,3} 或 {1,2}∪{3,4}，两者各 6 条边等权等概率，但第二种含 9 种等价组合 → 其逻辑错误的总概率更高，两匹配分属不同逻辑陪集。这是把解码目标从"物理单链"改为"逻辑陪集"的直接动机：
```
E = s(E)·t(s)·l(E)              # 任意错误三分量分解
p(L|s) ∝ Σ_{E:l(E)=L} p(E)      # 陪集后验 = 对所有稳定子变形求和
# 精确求解指数复杂（|S| 随稳定子数指数增长，且解码 NP-hard）
# 本论文近似：聚类划分稳定子群(B_c) → K 次优先级采样 → 投票
p̃(L_i|s) = n_{L_i}/K;  Ê = argmax_{L_i} n_{L_i}/K   # Lemma 1/2, Eq.12
```
Lemma 1 证明：K 个候选中逻辑错误相同的 E_i 互为退化错误、属同一逻辑等价陪集；Lemma 2 证明聚类把全局 coset ML 松弛为聚类内局部最优（B_c 位串空间，仅簇内 nontrivial syndrome 可激活稳定子变形）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
文献中简并性在稀疏/qLDPC 码中讨论最多（Fuentes et al., IEEE Access 2021 系统研究其解码影响）；处理方式有：Tensor-Network 收缩精确解 coset ML（精度高但收缩复杂度高）、BP 类消息传递 + 后处理、以及本论文的采样-投票近似（多项式时间，介于 UF 与 MWPM 之间）。使用时注意：简并性使"物理错误率"与"逻辑错误率"两个指标分离，评估解码器应以 LER/系统保真度为准而非物理匹配最优性。

涉及论文标题：
- Coset Ensemble Decoder for Quantum Error Correction with Algorithm-Hardware Co-Design

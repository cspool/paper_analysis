## Learned Index Structure for Neural Routing（基于学习索引的神经路由）

术语是什么？
Learned Index Structure 是 Kraska et al. (2018) 提出的一种数据库索引替代方案：使用机器学习模型（如 B-Tree 的神经网络替代品）来预测数据位置，而非使用传统数据结构（如 B-Tree、Hash Table）。PEER 论文声称首次将 learned index structure 应用于 MoE 路由——product key 可视为一种可学习的索引结构：两组子密钥 C, C' 是可学习的参数，查询网络 q 学习如何将输入映射到最相关 expert 对应的索引位置。与传统数据库索引不同，这里的"索引"不仅要考虑查找效率，还要根据输入语义选择最优 expert。PEER 路由器（product key + query network）的复杂度为 O(√N)——亚线性于 expert 数量 N，与可学习索引的目标（替代 O(log N) 的传统索引）一致。

从算法pipeline角度拆解术语：
Product Key 作为 learned index 的工作方式：
```
传统 MoE router（Token Choice / Expert Choice）：
    类似暴力扫描：计算所有 N 个 expert 的 score → TopK
    复杂度: O(N) — "无索引"，必须扫描所有 expert

Hash-based MoE router（Hash Layers, MoWE）：
    类似 Hash 索引：hash(token) → expert index
    复杂度: O(1) — 但 hash 函数固定（不可学习）

PEER router（Product Key）：
    类似 Learned Index：训练子密钥 C, C' 和 query net q
    q(x) 预测 x 在 product key 空间中的位置 → 检索附近 expert
    复杂度: O(√N) — 可学习的亚线性索引
```

术语一般如何实现？
Product Key 本身尚未在数据库领域作为 learned index 的标准实现；PEER 论文首次将其用于神经网络路由。子密钥 C, C' 存储为可学习参数（Embedding 矩阵），通过梯度下降端到端训练。与数据库 learned index 的区别：目标不是最小化查找延迟，而是在高维语义空间中检索语义相关的 expert。未来可能的扩展方向：更复杂的 learned index（如 RMI, PGM Index）应用于更大规模 expert 池。

涉及论文标题：
- Mixture of A Million Experts

---

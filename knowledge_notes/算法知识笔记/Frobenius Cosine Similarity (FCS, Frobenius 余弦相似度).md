## Frobenius Cosine Similarity (FCS, Frobenius 余弦相似度)

术语是什么？
标准余弦相似度在矩阵空间的推广。对于 U,V∈R^{h×h}：FrobCosSim(U,V) = ⟨Corr(U), Corr(V)⟩_F / (||Corr(U)||_F · ||Corr(V)||_F)。其中 Frobenius 内积 ⟨A,B⟩_F = trace(A^T·B)，Corr(U)_ij = U_ij/√(U_ii·U_jj)（消除尺度影响）。在 MoA 中用于衡量不同 agent 输出的语义相似度：将 last-layer hidden states T∈R^{n×h} 转为特征维相关矩阵 U=T^T·T∈R^{h×h}，再计算 FCS——解决了不同长度输出（n 不同）之间的相似度比较问题（通过 T^T×T 折叠 token 维）。

从算法pipeline角度拆解：
```
T_i = Embed(O_i) ∈ R^{n_i×h}    // 第 i 个 agent 的输出嵌入
T_j = Embed(O_j) ∈ R^{n_j×h}    // 可能不同长度

U = T_i^T × T_i ∈ R^{h×h}       // 折叠 token 维，保留特征维
V = T_j^T × T_j ∈ R^{h×h}

// 去尺度化
Corr(U)_ij = U_ij / sqrt(U_ii * U_jj)

// Frobenius 余弦
FCS = trace(Corr(U)^T · Corr(V)) / (||Corr(U)||_F · ||Corr(V)||_F)
```
选择特征维相关矩阵而非 token 维的原因：h（hidden dim）对所有输出固定，n（token 数）随输出长度变化——T^T×T 将可变维度折叠为固定 h×h。

术语一般如何实现？如何使用？
- 依赖共享 embedding model（如 Qwen3-Embedding-4B）确保跨模型可比
- 标准线性代数操作，高效实现
- 适用：多模型输出一致性评估、ensemble diversity 度量

涉及论文标题：
- Efficient Mixture-of-Agents Serving via Tree-Structured Routing, Adaptive Pruning, and Dependency-Aware Prefill-Decode Overlap

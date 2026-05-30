## Receptive Field in LLMs (LLM中的感受野)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

感受野（Receptive Field）在 LLM 语境下由 PowerAttention 论文首次系统定义：**模型在生成输出 token 时可利用的最大上下文 token 集合**。不同于 CV 中的空间感受野概念，LLM 中的感受野通过跨层信息传播建模——在层 l，token i 通过自注意力从单层感受野 A_i 中的 token 接收信息，经 FFN 处理后传播到下一层。在多层 LLM 中，token x 的感受野不仅包括直接关注的 token，还扩展到所有从 x 出发通过 DAG 可达的节点。形式化定义：给定 DAG G=(V,E)（V 为 token 节点，E 为注意力边集），token i 在 k 步内的感受野 R_k(i) = {j ∈ V | 从 i 到 j 的最短路径长度 ≤ k}。

感受野的两个关键属性：(1) 完整性（Completeness）——感受野是否能覆盖所有位置的 token；(2) 扩展效率（Expansion Efficiency）——感受野随层数的增长速度。PowerAttention 论文证明：即使不同稀疏模式在相同稀疏度下有相同的单层感受野大小（out-degree），设计良好的模式可以在多层传播后实现指数级增长的感受野。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**感受野的理论建模（PowerAttention 定理 B.1）**：

对于有向无环图 G，顶点标签 1..n，边集 E = {(i, j) | i-j = 2^k, k ∈ Z*}：
- 性质 1：任意顶点 i 的出度 < log n（每个 k 最多一条出边）
- 性质 2：任意顶点对 (i, j)，j < i，距离 ≤ log n

证明构造：距离 d = i-j 的二进制表示中为 1 的位为 k_1, k_2, ..., k_m（最多 log n 个），路径为：

$$i \to (i - 2^{k_1}) \to (i - 2^{k_1} - 2^{k_2}) \to \cdots \to j$$

路径长度 = popcount(d) ≤ log n。

**感受野的量化评估方法（Appendix A）**：

```
# 理论精度计算
B_k = 从最后 block 出发 k 步内可达的 block 集合
hat_alpha_k = |B_k| / 总 block 数  # 理论精度上界

# 实验精度计算（通过 passkey retrieval）
# 在 block 的范围内均匀采样 passkey 位置
alpha_k = B_k 内成功检索的样本数 / 总样本数  # 实验精度

# 关系：hat_alpha_k 是 alpha_k 的最小上界
```

**各模式感受野对比（Qwen2-7B, 32K context, 28 layers）**：

| 模式 | 6 层覆盖率 | 全覆盖所需层数 | 完整性 |
|------|-----------|---------------|--------|
| Sliding Window | ~14K / 32K | O(N) ≈ 14 layers | 完整 |
| Stride Slash | ~32K / 32K | O(√N) ≈ 6 layers | 完整 |
| Dilated | ~16K / 32K | N/A | ~50%（奇数位不可达） |
| LongNet | ~32K / 32K | O(log N) ≈ 5 layers | 不完整（段尾盲区） |
| PowerAttention | ~32K / 32K | O(log N) ≤ 5 layers | 完整 |

术语一般如何实现？如何使用？

感受野概念用于指导稀疏注意力模式的设计和评估：(1) 设计阶段——以最大化多层可达性为目标，在固定 out-degree（sparsity）约束下构造边集；(2) 评估阶段——通过 passkey retrieval 实验验证理论感受野与实际信息检索能力的一致性（PowerAttention Figure 1b 展示了理论和实验感受野的高度吻合）；(3) 调试阶段——通过 probing 分析（每层每位置训练 logistic classifier）可视化信息流，定位感受野覆盖盲区。PowerAttention 论文的 probing 实验揭示：即使 Full Attention 理论上单步可达所有 token，实际注意力头仍展示空间局部性——不仅检索 passkey 原始位置，还聚合相邻位置积累的信息。

涉及论文标题：
- PowerAttention: Exponentially Scaling of Receptive Fields for Effective Sparse Attention

---

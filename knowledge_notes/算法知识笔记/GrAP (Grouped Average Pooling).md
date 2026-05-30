## GrAP (Grouped Average Pooling)

术语解释
GrAP 是 LocMoE/ETR 论文提出的用于 MoE 路由层的新型特征提取层，替代传统的全连接 MLP Router。GrAP 按 expert 数量 n 对输入 hidden dimension d 进行分组平均池化，生成对角稀疏亲和力矩阵 W_aff ∈ R^{d×n}，参数仅为传统 Router 的 1/n.

术语是什么？
传统 MLP Router W_g ∈ R^{d×n} 是稠密参数矩阵，计算和存储开销为 O(d²)，且随机初始化的权重向量 w_i 之间不正交，容易导致多个 expert 学到相似路由模式（expert homogenization）。GrAP 将 d 维 token hidden state 按 expert 数 n 均分为 n 组，每组内取平均值，构成对角稀疏 W_aff：

$$W_{\text{aff}} = \operatorname{diag}(w_1, w_2, \ldots, w_n), \quad w_i[j] = \frac{n}{d} \text{ for } j \in [\frac{i \cdot d}{n}, \frac{(i+1) \cdot d}{n})$$

每个 w_i 仅在其分组内非零，不同 w_i 天然正交 (w_i · w_j = 0 for i ≠ j)。然后通过 cosine similarity δ_{t,i} = cos(x_t, w_i) 计算 token-expert 亲和力分数。

从算法pipeline角度拆解：
```
Input: x ∈ R^{s×d} (s tokens, d hidden dim)
Output: δ ∈ R^{s×n} (affinity scores)

# GrAP 前向: 对角稀疏，仅 d 个非零参数
W_aff = zeros(d, n)
for i in range(n):
    start = i * d // n
    end = (i+1) * d // n
    W_aff[start:end, i] = n / d

# 亲和力分数 (cosine similarity)
x_norm = L2_normalize(x, dim=-1)       # O(sd)
w_norm = L2_normalize(W_aff, dim=0)    # O(d)
delta = x_norm @ w_norm                # O(sd), 等价于分组平均池化 + cosine
```
对比传统 MLP Router: W_g ∈ R^{d×n} 全参数矩阵, O(s·d·n) 计算。GrAP 仅 O(s·d)，参数量降为 1/n。

术语一般如何实现？如何使用？
GrAP 在华为 MindSpeed-LLM 框架中实现，运行于 Ascend NPU 的 AI VECTOR CORE（cosine similarity 是向量操作）。分组数通常等于 expert 数 n。正交 gating 权重将 token space 按角度划分为 n 个扇区，每个 expert 对应一个扇区——等价于隐式 spherical k-means 聚类。

涉及论文标题：
- Expert-Token Resonance Redefining MoE Routing through Affinity-Driven Active Selection

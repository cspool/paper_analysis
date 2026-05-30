## Gated Linear Attention (GLA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Gated Linear Attention (GLA) 是由 Yang et al. (2024, ICML 2024) 提出的线性注意力变体，通过数据依赖的 2D 遗忘门增强标准线性注意力的表达能力。核心公式：S_t = G_t ⊙ S_{t-1} + K_t^T V_t，其中 G_t ∈ R^{d×d} 是输入依赖的门控矩阵（在 channel 和 head 两个维度上运作）。相比标准线性注意力（等权累积），GLA 的门控允许模型选择性"遗忘"不相关信息。配套硬件高效实现为 FlashLinearAttention（FLA），使用分块策略：块内矩阵乘法（利用 Tensor Core），块间递归传递状态。

在 SAMBA 中：Sliding GLA 替换 Samba 架构中的 Mamba 层进行消融（Table 3）。438M 规模上 GLA 的 perplexity（10.43/10.00/9.92 at 4K/8K/16K）优于 Mamba（10.70/10.30/10.24）但不如 Samba（10.06/9.65/9.57）。GLA 训练速度（4.94×10^5 tokens/s）显著快于 Mamba（2.46×10^5），因为 Mamba 层数更多且 scan kernel 开销大。GLA 加入短卷积改善不明显（10.43→10.39），因已有 channel 级细粒度门控。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
# GLA 核心递归（简化）
S_0 = 0  # [d, d]
for t in 1..n:
    G_t = sigmoid(Linear_g(X_t))  # 输入依赖的门控
    S_t = diag(G_t) @ S_{t-1} + K_t^T @ V_t   # [d, d]
    o_t = Q_t @ S_t                             # [d]
```
训练时使用分块并行化：chunk 内矩阵乘法并行，chunk 间递归传递 S_t。FlashLinearAttention 库提供 Triton 实现。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：Flash Linear Attention (FLA) 库（https://github.com/sustcsonglin/flash-linear-attention），Triton kernel。GLA 在 1.3B 规模保持良好长度外推：训练于 4K，16K 时 perplexity=7.19（优于 Mamba 7.15，弱于 Samba 6.96，Table 3）。相比 Mamba：训练速度更快（不需要复杂 scan kernel）、可与 SWA 直接组合；下游任务略弱于 Mamba-based 混合模型。适用场景：需要训练速度优先于极致下游性能的大规模 LM 预训练。

涉及论文标题：
- Samba__Simple_Hybrid_State_Space_Models_for_Efficient_Unlimited_Context_Language_Modeling


---

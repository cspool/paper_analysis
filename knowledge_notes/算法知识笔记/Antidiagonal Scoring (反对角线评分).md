## Antidiagonal Scoring (反对角线评分)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Antidiagonal Scoring（反对角线评分）是 XAttention 论文提出的 block 重要性估计方法。核心洞察：注意力矩阵的**反对角线值之和**（antidiagonal sum）可以作为 block 重要性的高效代理指标。反对角线指从矩阵左下到右上的对角线方向（即方向为 $(-1, +1)$ 的线），与标准主对角线正交。

与现有方法的对比：(1) MInference/FlexPrefill 使用 mean/sum pooling 估计 block 重要性——但 pooling 在 block 内仅有少量显著垂直/斜线模式时会严重低估重要性；(2) 反对角线天然交叉 block 内所有可能的垂直列和斜线模式（见 Vertical-Slash Attention Pattern），确保不遗漏任何关键模式；(3) 每个 token 至少参与一条反对角线，保证信息完整性。

Strided 变体（Strided Antidiagonal Scoring）：以步长 S 在 B×B 的 block 内沿反对角线采样，将 Q reshape 为 S 组、K reshape 为 S 组，计算 S×S 的近似注意力矩阵。计算复杂度仅为完整注意力的 $1/S^2$。

从算法pipeline角度拆解术语：

```
# Strided Antidiagonal Scoring（block size B, stride S）
Input: Q, K ∈ R^{L×d}, block_size B, stride S
Output: block importance scores

# 对每个 B×B 的 attention block
for b = 0 to N_B - 1:
    # Step 1: Q anti-diagonal reshape [B, d] -> [S, B//S, d]
    Q_reshaped = []
    for i = S-1 down to 0:  # 从下到上遍历反对角线
        Q_reshaped.append(Q[b*B:(b+1)*B, :][i::S, :])

    # Step 2: K stride reshape [L, d] -> [S, L//S, d]
    K_reshaped = []
    for i = 0 to S-1:
        K_reshaped.append(K[i::S, :])

    # Step 3: 近似注意力分数
    A_approx = Softmax(Q_reshaped @ K_reshaped^T / sqrt(d_h) / S)

    # Step 4: 反对角线分数 = block 重要性
    score[b] = sum of antidiagonal values in A_approx
```

关键性质：反对角线交叉每个 block 内所有可能的垂直和斜线模式（见 XAttention Figure 2）。即使 block 内仅有一条垂直列或斜线，反对角线也必定与之相交。

消融实验（Table 6）：同等计算量下，antidiagonal 模式比 random 和 diagonal 模式密度更低且准确率更高——S=8 时 antidiagonal avg 88.47 (density 20.97%) vs random 82.48 (27.57%) vs diagonal 81.06 (24.47%)。

术语一般如何实现？如何使用？

基于 FlashInfer 框架实现。反对角线 scoring 作为 prefill attention 的预处理步骤——在 FlashAttention kernel 调用前，先执行轻量级 Q/K reshape + 小矩阵乘法来估计 block 重要性。计算开销极低（O(L×d/S²)），仅占总 prefill 时间的很小比例。

使用场景：适用于任意 Transformer 模型的 attention 模块（causal 和 non-causal 均支持），作为 block-sparse attention 的 block selection 指导。已在 Llama-3.1-8B（文本）、Qwen2-VL-7B（视频理解）、HunyuanVideo（视频生成/DiT）上验证。Stride S 的选择需权衡——S 越大越稀疏但可能漏检斜线模式（S=64 时 avg 降至 81.21）；推荐 S=8 或 S=16。

涉及论文标题：
- XAttention: Block Sparse Attention with Antidiagonal Scoring

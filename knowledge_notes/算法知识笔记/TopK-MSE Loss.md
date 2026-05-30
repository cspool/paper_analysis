## TopK-MSE Loss

术语解释
TopK-MSE Loss 是 EAC-MoE 中用于校准 MoE router 的损失函数，在计算 router 输出的 MSE 时，仅对 top-K 最高概率的 expert 计算损失，而非所有 N 个 expert。其核心动机是：量化后 shifted expert（全精度选中但量化后未选中）95.9% 仍排名在 top-16 概率内（64 expert 中），但 top-16 的 MSE 损失仅占全部 N 个 expert MSE 损失的 29.25%。如果对所有 expert 计算 MSE，损失会被大量低概率 expert 的噪声主导，优化过程难以聚焦于真正重要的 expert 对齐。

术语是什么？
TopK-MSE Loss 的公式：

$$\mathcal{L} = \frac{1}{K} \sum_{i \in \mathrm{top-}K(W_r x)} ((W_r x)_i - (W_r \hat{x})_i)^2$$

其中 $W_r$ 是 router 权重矩阵，$x$ 是全精度模型输入，$\hat{x}$ 是量化后模型的输入（经过量化 MHSA 及已量化 expert 的激活值）。通过仅对全精度 router 输出的 top-K 高的 expert 计算损失，优化器聚焦于对齐"更可能被选中的 expert"的 router 输出。

K 值通过网格搜索确定：对 Phi3.5-moe（16 expert, top-2）K=8；对 Deepseek-moe-16b-base（64 expert, top-6）K=20；对 Qwen1.5-MoE-A2.7B（60+4 expert, top-4）K=20。K 值过小（接近 per-token selection count）会过拟合；过大则退化为全量 MSE（噪声主导）。

从算法pipeline角度拆解术语：
```
=== TopK-MSE Loss 计算与 Router 校准 ===
输入: router权重 W_r, 全精度输入 x, 量化后输入 x_hat, 超参数 K
输出: 校准后的 router 权重

# 1. 前向计算
logits_full = W_r @ x        # [num_experts], 全精度参考
logits_quant = W_r @ x_hat   # [num_experts], 量化后实际输出

# 2. 确定 top-K 索引（基于全精度参考，非量化后）
topK_indices = arg_top_k(logits_full, K)  # 仅关注最重要 K 个 expert

# 3. TopK-MSE Loss
loss = 0
for i in topK_indices:
    loss += (logits_full[i] - logits_quant[i])^2
loss = loss / K

# 4. 反向传播更新 W_r，对齐量化后 router 输出与全精度参考
# 低概率 expert 的偏差被忽略，避免噪声主导优化
```

术语一般如何实现？如何使用？
- 在逐层量化校准过程中使用：量化每层 MHSA 后计算 TopK-MSE Loss 校准该层的所有 MoE router
- K 值选择需网格搜索（在 MMLU 上评估不同 K 值的效果）
- 相比全量 MSE Loss：Phi3.5-moe 2.06-bit 下准确率 65.03%（TopK-MSE）vs 64.52%（MSE）；Deepseek-moe 57.05% vs 55.91%
- 高量化位宽（如 3.03-bit）时 K 值敏感性低（expert-shift rate 本身较低）
- 可与任何基于 GPTQ 的 MoE 量化方法正交结合

涉及论文标题：
- EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models

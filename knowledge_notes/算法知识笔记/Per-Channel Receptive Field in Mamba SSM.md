## Per-Channel Receptive Field in Mamba SSM

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Per-Channel Receptive Field是LongMamba提出的Mamba隐藏状态通道的可视化分析概念。利用Mamba注意力分数α_{i,j} = C_i^T (∏_{k=j+1}^i Ā_k) ⊙ B̄_j ∈ R^{d_e}（Ali et al., 2024），该分数是第j个token对第i个token输出的per-channel贡献。LongMamba在log scale下可视化α_{i,j}矩阵，用红色边框标记attention score > 10^{-3}的范围作为通道的"感受野"。分析发现Mamba的不同通道具有截然不同的感受野长度——有些仅关注~200 tokens的局部上下文，有些覆盖整个训练序列长度(~2000)。在Mamba-130M第12层48个通道的可视化中，感受野排序显示清晰的二分结构——累积衰减小者拥有全局感受野。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
α_{i,j}[c] = C_i[c]^T · (∏_{k=j+1}^i Ā_k[c]) ⊙ B̄_j[c]  # 标量, per-channel

# 通道c的感受野（序列末尾token L的视角）:
receptive_field[c] = argmin_j { j | α_{L,j}[c] > 10^{-3} }
# 短→local channel;  长（≈L）→ global channel

# LongMamba Step 1: 通过累积衰减而非可视化分类:
decay_c = ∏_{k=1}^L Ā_k[c]  # 沿d_s维度取平均
channel_c = "global" if decay_c > θ else "local"
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
使用Pile采样序列进行分析，θ通过LongBench-E上grid search确定（候选10^{-40}到5×10^{-1}）。不同模型的最优θ差异巨大（Mamba-1.4B: 10^{-30}, Mamba2-1.3B: 5×10^{-2}, Zamba2-1.2B: 10^{-5}）。全局通道需要token filtering来扩大感受野。代码：https://github.com/GATECH-EIC/LongMamba。

涉及论文标题：
- LongMamba__Enhancing_Mamba_s_Long_Context_Capabilities_via_Training-Free_Receptive_Field_Enlargement

---

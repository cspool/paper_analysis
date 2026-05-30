## GroupNorm in Linear Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GroupNorm (Wu & He, 2018) 被 SUPRA 创新性地用于线性注意力的输出归一化，替代传统分母除法。传统线性注意力 v'_i = Σ sim·v / Σ sim 的分母可能数值发散或趋零。SUPRA 用 GroupNorm(num_groups=num_heads) 在每个 head 的 WKV 输出上做独立归一化（减均值除标准差），实现：(1) 数值稳定；(2) 无需维护额外归一化状态；(3) 保持 head 独立性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 传统线性注意力（不稳定）:
v'_i = (Σ sim·v) / (Σ sim)  # 分母→0 时 NaN

# SUPRA GroupNorm（稳定）:
wkv_i = Σ γ^{i-j}·sim(q_i,k_j)·v_j  # (B, h, seq, d_h)
v'_i = GroupNorm(h)(wkv_i)  # h=num_heads, 每head独立归一化
# mean_h = mean(wkv_i along d_h), std_h = std(wkv_i along d_h)
```

Table 3 消融证明：T2R（分母除法）在 1B uptraining 时 HellaSwag 40.6（vs SUPRA 57.0），归一化策略是大规模 uptraining 的关键。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch 实现：`nn.GroupNorm(num_groups=num_heads, num_channels=D)`。group_size = d_h = D/h，典型值 64-128，足够保证统计量精度。

涉及论文标题：
- Linearizing_Large_Language_Models

---

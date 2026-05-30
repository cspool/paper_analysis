## Grouped-Tied Attention (GTA / 分组绑定注意力) [Zadouri et al., 2025]

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Grouped-Tied Attention (GTA) 是 Zadouri、Strauss 和 Dao（Princeton University, 2025）提出的一种硬件高效注意力变体。核心思想是将 GQA（Grouped-Query Attention）中的独立 Key 和 Value 投影**绑定为单一状态**（tied KV state），将 KV cache 大小减半、算术强度翻倍，同时保持与 GQA 相当或更优的模型质量。

三个关键设计要素：(1) **KV Tying**：单一投影矩阵 W^{KV} 替代独立的 W^K 和 W^V，输出 shared *tied KV* 向量；(2) **Partial RoPE**：仅前半 head 维度（d_h/2）作为无位置编码的 content key（K_NoPE），后半来自独立的单头 RoPE 投影（K_RoPE，跨所有 KV head 广播）；(3) **Full value dimension**：value 路径使用 tied KV 的完整 d_h 维，保证 value 表达力不受损。

KV cache per token：hkv × 1.5 × d_h（含 d_h/2 广播 RoPE），vs GQA 的 hkv × 2 × d_h。算术强度约 2gq vs GQA 的 ~gq。

**注意**：本 GTA 与 Sun et al., 2025 的 "Grouped-head latenT Attention (GTA)" 是**不同方法**，仅共享缩写。Sun 的 GTA 使用共享 attention map + 非线性 value decoder；本 GTA 使用 KV 绑定 + 部分 RoPE。

从算法pipeline角度拆解术语，给出具体例子。

```
# GTA-4: hq=16, hkv=4, gq=4, dh=128, d_R=64
# 输入: X ∈ R^{B×L×d}

# 1. Q 投影（标准）
Q = X @ W^Q              # [B, L, 16, 128]

# 2. Tied KV 投影 —— 单一投影替代 W^K + W^V
KV = X @ W^{KV}          # [B, L, 4, 128] —— tied state

# 3. 构造 K 和 V
V = KV                   # value 用完整维度
K_NoPE = KV[:,:,:,:64]  # 前半维度，不加 RoPE
K_RoPE = apply_rope(X @ W^{K_RoPE})  # [B, L, 1, 64]
K_RoPE = broadcast(K_RoPE, 4)        # 广播到 4 个 KV head
K = concat([K_NoPE, K_RoPE], dim=-1)  # [B, L, 4, 128]

# 4. GQA-style attention
for g in 0..3:
    Q_g = Q[:, :, g*4:g*4+4, :]      # 4 query heads
    attn[g] = softmax(Q_g @ K_g^T / 8) @ V_g
```

**KV cache 对比（XL 1.471B, hq=16, hkv=4, dh=128, BF16）**：

| 方法 | bytes/token TP=1 | bytes/token TP=2 | bytes/token TP=4 |
|------|------------------|------------------|------------------|
| GQA-4 | 2048 | 1024 | 512 |
| GTA-4 | 1152 | 640 | 384 |
| MLA | 1152 | 1152 | 1152 |
| GLA-2 | 1152 | 640 | 640 |

GTA 在低 TP 度下比 GQA 节省最多（TP=2: 640 vs 1024），且随 TP 增加持续缩小 per-device cache。

术语一般如何实现？如何使用？

在 PyTorch 中替换标准 MHA 模块的 W_K 和 W_V 为 W^{KV}（输出 hkv×dh）+ W^{K_RoPE}（输出 1×dh/2）。训练配置与 baseline 一致（AdamW, cosine LR, FineWeb-Edu-100B），各 variant 通过加宽 FFN 匹配 MHA 参数总量。推理 kernel 开源：https://github.com/Dao-AILab/grouped-latent-attention，包含 warp specialization + software pipelining 优化。

适用场景：需要比 GQA 更小 KV cache、但保持分片能力的低 TP 度分布式推理。

涉及论文标题：
- Hardware-Efficient_Attention_for_Fast_Decoding

---

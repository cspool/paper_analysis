## Non-uniform Transformer Block Architecture (非均匀 Transformer 块架构)

术语解释
Non-uniform Transformer Block Architecture 是打破标准 Transformer 中 attention 和 FFN 严格交替排列的架构设计范式，允许 block 内以任意顺序排列多种层类型（attention、dense FFN、sparsely gated FFN/MoE），通过自动搜索或手动设计优化层序列和维度配置。

术语是什么？
标准 Transformer (Vaswani et al., 2017): 每层 = Attention + FFN, uniform 重复。GLaM 引入稀疏性但保持交替：每层 = Attention+FFN 或 Attention+MoE, 交替重复。Non-uniform 架构完全打破此约束，一个 block 可包含多个 sub-layer 且顺序不受限：

$$\mathcal{N} = \mathcal{F}_k \odot ... \odot \mathcal{F}_2 \odot \mathcal{F}_1(X_1) = \bigcup_{j=1...k} \mathcal{F}_j(X_1)$$

其中 $\mathcal{F}_i \in \{\mathcal{F}_{\text{attn}}, \mathcal{F}_{\text{moe}}, \mathcal{F}_{\text{ffn}}\}$。

从算法pipeline角度拆解术语。
以 Brainformer Block 1 的 forward 为例（8 sub-layers, 搜索得到的最优非均匀架构）：

```
# Brainformer Block 1: 8 sub-layers 非均匀组合
# 架构: Attn → MoE(EC) → FFN → Attn → MoE(EC) → FFN → MoE(EC) → FFN
# 特征: Attention 仅 2/8 (vs uniform 4/8), MoE 3/8, FFN 3/8

def brainformer_block_forward(X_1):  # [B, L, d=1024]
    X = X_1
    X = X + MultiHeadAttention(LayerNorm(X))        # SL1: Attn, 20 heads
    X = X + ExpertChoiceMoE(LayerNorm(X))           # SL2: MoE, 64E, cap=1
    X = X + FFN_GeLU(LayerNorm(X))                  # SL3: FFN, hidden=2048
    X = X + MultiHeadAttention(LayerNorm(X))        # SL4: Attn
    X = X + ExpertChoiceMoE(LayerNorm(X))           # SL5: MoE
    X = X + FFN_GeLU(LayerNorm(X))                  # SL6: FFN
    X = X + ExpertChoiceMoE(LayerNorm(X))           # SL7: MoE
    X = X + FFN_GeLU(LayerNorm(X))                  # SL8: FFN
    return X
```

术语一般如何实现？如何使用？
- 通过 Evolutionary Search 自动发现最优 sub-layer 序列
- Ablation 发现：层类型比例（attention:MoE:FFN ratio）对质量至关重要，层顺序（order）相对不重要
- 搜索倾向于减少 attention 频率（attention 计算昂贵，尤其在长序列上）
- 搜索倾向于增大 model dim 同时减小 expansion ratio（利用 MoE 多 expert 替代单层大 FFN）
- 相关工作：Sandwich Transformer (Press et al., 2019) 重排但保持 uniform；EfficientNet (Tan & Le, 2019) per-layer scaling for CNN

涉及论文标题：
- Brainformers Trading Simplicity for Efficiency

---

## Granularity-Bit Controller (GBC)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Granularity-Bit Controller (GBC) 是 Granular-DQ 的核心组件，对输入图像的每个 patch 进行粗到细的多粒度层次分析，自适应分配量化 bit-width。设计哲学：不同 patch 的贡献比例不同——细粒度特征揭示局部纹理复杂度，粗粒度特征表达整体场景结构——应根据贡献比例分配计算精度。

GBC 工作流程：(1) 编码器 $\mathcal{E}$ 对输入 X 提取 D 层多粒度特征 $\mathbf{Z} = \{Z_1, ..., Z_D\}$（D-1 次下采样，$Z_1$ 最细粒度，$Z_D$ 最粗粒度）；(2) 所有粒度特征 GroupNorm → 平均池化至 $Z_D$ 分辨率 → concat → GAP 得到通道统计量 $\mathbf{S}$；(3) 线性层 $\mathbf{W}_g \in \mathbb{R}^{(N \times D) \times N}$ 作用于 $\mathbf{S}$ 生成门控 logits $\mathbf{G}$；(4) Gumbel-Softmax 为每个 patch 采样门控分数 $p_i$（patch 贡献比例），映射到候选 bit code {4,6,8}。

从算法pipeline角度拆解术语，给出术语所在pipeline的伪代码或具体计算过程。

```
# GBC 伪代码
输入: 图像 X, 编码器 E, D 层多粒度

# 多粒度特征提取与融合
Z = E(X)  # Z = [Z_1,...,Z_D], Z_1 最细粒度, Z_D 最粗粒度
Z_hat = [GroupNorm(Z_d) → AvgPool(Z_D_res) for Z_d in Z]
Z_cat = concat(Z_hat, dim=channel)

# 通道统计量 + Bit 分配
S = GlobalAvgPool(Z_cat)
G = Linear(W_g)(S)  # W_g: (D×C) → N (N=3, bit codes [4,6,8])

for each patch X_i:
    σ = sample_gumbel(N)
    θ_i = argmax(G[i] + σ)        # 离散门控索引
    p_i = softmax((G[i]+σ)/τ)[θ_i] # 门控分数
    b_i = bit_codes[θ_i]           # → {4, 6, 8}

输出: 每个 patch 的 bit-width b_i (所有层共享)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

GBC 使用 PyTorch 的 `F.gumbel_softmax` 实现可微分离散采样。训练时端到端优化（仅 L1 loss），推理时直接 argmax 确定 bit-width。GBC 置于 SR 网络最前端，对任何 CNN/Transformer SR 架构即插即用。各层对同一 patch 使用相同 bit-width（layer-invariant），避免 CADyQ 逐层 bit selector 对层间关系的破坏。

涉及论文标题：
- Thinking in Granularity Dynamic Quantization for Image Super-Resolution by Intriguing Multi-Granularity Clues

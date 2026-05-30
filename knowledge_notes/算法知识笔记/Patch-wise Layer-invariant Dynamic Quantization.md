## Patch-wise Layer-invariant Dynamic Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Patch-wise Layer-invariant Dynamic Quantization 是 Granular-DQ 提出的新型动态量化范式。与 CADyQ/CABM 同时逐层（layer-wise）和逐块（patch-wise）自适应 bit-width 不同，Granular-DQ 仅针对 patch 自适应——同一图像的不同 patch 可能获得不同 bit-width，但 SR 网络所有层对同一 patch 使用相同 bit-width。

"Layer-invariant" 的核心洞察：逐层独立调整 bit-width 会扰乱原始模型各层间的表示关系。论文通过 t-SNE 可视化证实——CADyQ 量化后各层特征分布明显更离散（与原始全精度模型偏差大），而 Granular-DQ 特征分布更接近原始模型。这解释了为什么放弃 layer sensitivity 反而获得更好量化效果。

从算法pipeline角度拆解术语：

```
# Granular-DQ: Patch-wise + Layer-invariant
b_i = GBC+E2B(X_i)  # 每个 patch 一个全局 bit-width
for conv_layer L_k in SR_model:
    for each patch X_i:
        X_i_hat = Q_{b_i}(X_i)         # 同一 b_i 贯穿所有层
        X_i = conv2d(X_i_hat, Q_8(W_k))
# 优势: 层间关系不被打乱

# CADyQ: Layer-wise + Patch-wise (对比)
for conv_layer L_k:
    for each patch X_i:
        b_{k,i} = BitSelector(L_k, X_i)  # 每层独立选择
        X_i = Q_{b_{k,i}}(conv2d(X_i, W_k))
# 问题: b_{k,i} 随层变化, 破坏原始层间关系
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

该范式通过 GBC 集中决策 bit-width，避免每层插入 bit selector 的计算开销。量化器使用 QuantSR（可替换为 PAMS），权重固定 8-bit。Transformer attention block 保持全精度（因量化误差过大）。训练仅需 L1 loss，无需 KD loss 或 bit 正则化项。局限性：混合精度方案需特定硬件和算子支持才能实现真正的压缩加速。

涉及论文标题：
- Thinking in Granularity Dynamic Quantization for Image Super-Resolution by Intriguing Multi-Granularity Clues

---

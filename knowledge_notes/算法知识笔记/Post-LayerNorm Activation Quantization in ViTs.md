## Post-LayerNorm Activation Quantization in ViTs

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Post-LayerNorm 激活指 ViT Transformer Block 中 Layer Normalization 后的激活值。关键特性：严重的通道间变异性（severe inter-channel variation）——不同通道的数值分布差异巨大。这种高变异性使粗粒度 layer-wise 量化产生大量化误差。I&S-ViT 首次系统分析了不同量化粒度对 loss landscape 的影响：全精度权重 + channel-wise 激活量化产生平滑低 loss landscape，而 channel-wise 权重量化 + layer-wise 激活量化的 landscape 粗糙且高 loss。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
I&S-ViT SOS 策略对 post-LayerNorm 激活的处理：
```
# Stage1: channel-wise量化 (每通道独立s_c, z_c)
# → 平滑loss landscape, 稳定优化
# Stage2: Scale Reparameterization → layer-wise
# → 无损转换为高效layer-wise方案
# Stage3: layer-wise量化 + 量化权重微调
# → 恢复权重量化损失
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
RepQ-ViT 在离线校准后通过 scale reparameterization 将 channel-wise 量化器转为 layer-wise。I&S-ViT 扩展此方法到优化-based PTQ，在 SOS Stage1 利用 channel-wise 平滑 landscape 进行优化，Stage2 无损转换。DeiT-S W3A3：RepQ-ViT 仅 4.37%，I&S-ViT SOS 单独达 45.19%（+40.82%），证明优化-based 方法在低比特场景远远优于纯校准方法。

涉及论文标题：
- I&S-ViT: An Inclusive & Stable Method for Pushing the Limit of Post-Training ViTs Quantization

---

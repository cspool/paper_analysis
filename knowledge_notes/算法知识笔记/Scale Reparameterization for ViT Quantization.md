## Scale Reparameterization for ViT Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Scale Reparameterization 是将 channel-wise 量化器等价转换为 layer-wise 量化器的技术，源自 RepQ-ViT (ICCV 2023)。给定 channel-wise 参数 (s∈R^D, z∈R^D)，计算均值作为 layer-wise 参数 (s̃=Mean(s), z̃=Mean(z))，然后调整 LayerNorm affine 参数和下一层权重/偏置将 per-channel scale 差异"吸收"到网络参数中。公式：r₁=s/s̃, r₂=z-z̃; β̃=(β+s⊙r₂)/r₁, γ̃=γ/r₁; W̃_{:,j}=r₁⊙W_{:,j}, b̃_j=b_j-(s⊙r₂)·W_{:,j}。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
给定: post-LayerNorm激活 A∈R^{N×D}, channel-wise quantizer (s,z)

# Step 1: LayerNorm 重参数化
原始: A_i = γ_i*(X_i-μ)/σ+β_i
       A_q_i = round((A_i-z_i)/s_i)   # channel-wise量化
替代: β̃_i = (β_i+s_i*(z_i-z̃))/r1_i   # r1_i=s_i/s̃
       γ̃_i = γ_i/r1_i
       Ã_q = round((Ã_i-z̃)/s̃)          # layer-wise量化(s̃,z̃为标量)
可证: Ã_q ≡ A_q_i (数学等价)

# Step 2: 下一层权重补偿
W̃_{:,j}=r₁⊙W_{:,j}; b̃_j=b_j-(s⊙(z-z̃))·W_{:,j}
可证: Ã_q·W̃+b̃ ≡ A_q·W+b (输出等价)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 PyTorch 中遍历所有 post-LayerNorm 层，直接修改 state_dict 中对应的 LayerNorm affine 参数和后续 FC 权重/偏置。I&S-ViT 在 SOS Stage 2 中权重仍为全精度时执行转换，确保无损（lossless transition），与 RepQ-ViT 在量化权重上的有损转换形成关键区别。

涉及论文标题：
- I&S-ViT: An Inclusive & Stable Method for Pushing the Limit of Post-Training ViTs Quantization

---

## Adaptive LayerNorm (AdaLN / 自适应层归一化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
AdaLN（Adaptive Layer Normalization）是 ARVG 模型和 DiT（Diffusion Transformer）中的核心模块，用于将 conditioning 信息（类别标签、时间步、位置编码等）动态注入到网络激活中。与标准 LayerNorm 不同，AdaLN 不仅做归一化，还通过 conditioning 生成 shift（β）和 scale（γ）参数来调整激活分布：output = LayerNorm(x) * γ(cond) + β(cond)。在 ARVG 中，AdaLN 的作用是保持预测 token 之间的双向依赖关系和条件引导——由于自回归生成中每个 token 只能看到之前的 token，双向依赖通过 conditioning 中的位置信息和类别信息来间接维护。AdaLN 通常在每个 block 中作用于 MHSA 之前和 FFN 之前：MHSA 输入 = LN(x) * MHSA_scale1(cond) + MHSA_shift1(cond)；FFN 输入 = LN(x_attn_out) * FFN_scale1(cond) + FFN_shift1(cond)。

从算法pipeline角度拆解术语，给出具体例子。
以 RAR 模型中一个 Transformer block 的 AdaLN 计算流程为例：

```
输入: x (隐层激活), cond (conditioning: 类别嵌入 + 位置嵌入)
输出: modulated_x

# AdaLN 生成调制参数 (通过线性投影)
MHSA_shift1, MHSA_scale1, MHSA_shift2, MHSA_scale2,
FFN_shift1, FFN_scale1, FFN_shift2, FFN_scale2 = AdaLN_proj(cond)
# 共 8 组参数，每组 shape: [T, C] (T=tokens, C=channels)

# === MHSA 路径 ===
x_norm = LayerNorm(x)
x_modulated = x_norm * (1 + MHSA_scale1) + MHSA_shift1   # 第一次调制
attn_out = MHSA(x_modulated)
attn_modulated = attn_out * (1 + MHSA_scale2) + MHSA_shift2  # 第二次调制
x = x + attn_modulated                                   # 残差连接

# === FFN 路径 ===
x_norm = LayerNorm(x)
x_modulated = x_norm * (1 + FFN_scale1) + FFN_shift1    # 第一次调制
ffn_out = FFN(x_modulated)                               # fc1 -> activation -> fc2
ffn_modulated = ffn_out * (1 + FFN_scale2) + FFN_shift2 # 第二次调制
x = x + ffn_modulated                                     # 残差连接
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
AdaLN 对量化的影响：(1) AdaLN 生成的 scale/shift 参数引入了高度动态的激活分布——位置嵌入沿 token 维度变化导致 AdaLN 输入在 token 维度上高度动态；(2) AdaLN 的 scale 参数可能产生 channel-wise outlier（某些通道的 scale 远大于其他通道），这是 PTQ4ARVG 识别的第一个关键挑战；(3) AdaLN 不保持旋转不变性——与标准 LayerNorm 的 RMSNorm(XQ) = RMSNorm(X)Q 性质不同，AdaLN 的 LN(X) * scale + shift 形式破坏了旋转不变性，使得 QuaRot 的 Hadamard 旋转矩阵无法离线融合，必须在线计算，导致额外推理开销。在 PTQ4ARVG 的 GPS 方法中，scaling factor 被融合到 AdaLN 权重中以实现零推理开销。DiT 模型也使用 AdaLN，ViDiT-Q 方法同样通过在线旋转处理其量化。

涉及论文标题：
- PTQ4ARVG Post-Training Quantization for AutoRegressive Visual Generation Models

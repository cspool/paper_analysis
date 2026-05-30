## ARVG (AutoRegressive Visual Generation / 自回归视觉生成)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ARVG（AutoRegressive Visual Generation）是一类将自回归生成范式应用于视觉生成任务的模型家族。受 LLM 中自回归生成的成功启发，ARVG 模型采用与 LLM 兼容的 Transformer 架构，通过逐 token 预测的方式生成图像。核心架构由 L 个 block 组成，每个 block 包含 Multi-Head Self-Attention（MHSA）、Feedforward Network（FFN）和 Adaptive LayerNorm（AdaLN）。AdaLN 将 conditioning 信息（包括类别标签和位置编码）转换为 shift 和 scale 参数来调整激活分布，从而保持生成 token 之间的双向依赖关系和条件引导。与 LLM 的关键区别：(1) ARVG 预测固定数量的 token（图像分辨率决定了 token 序列长度）；(2) ARVG 以条件信息作为初始 token（sink token）；(3) 不同 ARVG 模型在 token 预测粒度上有差异——VAR 一次预测一个 scale 的所有 token，RAR 一次生成一个 token，PAR 先顺序预测一个 token 再并行预测多个非局部 token，MAR 一次预测多个随机 token。代表模型包括 VAR（2B）、RAR（1.5B）、PAR（3B）、MAR（1B）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 VAR 模型的推理 pipeline 为例（一个 scale 的 token 生成）：

```
输入: 类别标签 c, 位置编码 p
输出: 图像 tokens

# Step 1: 编码 conditioning
cond = encode(c) + encode(p)  # 类别信息 + 位置信息

# Step 2: 初始化（sink token）
x_0 = cond  # 条件信息作为初始 token

# Step 3: 逐 block 处理（L 个 block）
for block l = 1 to L:
    # AdaLN: conditioning 调整激活分布
    shift_attn, scale_attn, shift_ffn, scale_ffn = AdaLN_l(cond)

    # Multi-Head Self-Attention
    x_norm = LayerNorm(x)
    x_scaled = x_norm * scale_attn + shift_attn          # AdaLN 调制
    attn_out = MHSA(Q=x_scaled, K=x_scaled, V=x_scaled)  # 自注意力
    x = x + attn_out                                      # 残差连接

    # Feedforward Network
    x_norm = LayerNorm(x)
    x_scaled = x_norm * scale_ffn + shift_ffn            # AdaLN 调制
    ffn_out = FFN(x_scaled)                               # 两层 MLP + 激活
    x = x + ffn_out                                       # 残差连接

# Step 4: 预测 next-scale tokens
tokens = output_head(x)  # 线性投影 + softmax
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
ARVG 模型使用与 LLM 兼容的 Transformer 架构，通常在 ImageNet 上训练，使用交叉熵损失进行 next-token prediction。推理时支持 KV Cache 加速，但由于 AdaLN 的存在，每个 block 的激活分布由 conditioning 动态调整，导致量化面临三大挑战：(1) channel-wise outlier（AdaLN 调制后的激活存在严重的通道间范围差异）；(2) token-wise 动态激活（位置嵌入随 token 位置变化，首 token 为含关键条件信息的 sink token）；(3) sample-wise 分布不匹配（跨样本的激活高度相似，导致校准冗余）。这些特点也是 PTQ4ARVG 论文的核心动机。ARVG 代表模型 VAR、RAR、PAR、MAR 的开源代码分别在其官方 GitHub 仓库中。

涉及论文标题：
- PTQ4ARVG Post-Training Quantization for AutoRegressive Visual Generation Models

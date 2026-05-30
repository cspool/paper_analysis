## Vision-Language Adaptor (视觉-语言适配器)

术语解释
Vision-Language Adaptor (VL Adaptor) 是 LLaVA-style VLM 中的连接模块，负责将 vision encoder 的视觉特征投影到 LLM 的文本 embedding 空间，实现视觉和语言两种模态的特征对齐。DeepSeek-VL2 的 VL Adaptor 由 2×2 pixel shuffle 压缩 + 2-layer MLP 组成。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
VL Adaptor 的核心功能：(1) 维度对齐：将 vision encoder output（如 SigLIP 的 1152/4608 dim）投影到 LLM embedding dimension（如 1280/2048/2560）；(2) Token 压缩：通过 pixel shuffle 将视觉 token 数减少 4×（27×27→14×14=196 tokens/tile），降低后续 LLM 的计算负担；(3) 模态桥接：MLP 学习从视觉特征空间到语言特征空间的非线性映射。在 DeepSeek-VL2 中，VL Adaptor 还负责插入 special tokens (<tile_newline>, <view_separator>) 来编码 tile 的 2D 空间结构信息。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
=== VL Adaptor Forward Pass (per tile) ===
Input: v ∈ R^{27×27×1152}  (SigLIP output per tile)

// Step 1: 2×2 Pixel Shuffle (token compression)
// Reshape spatial dimensions: 27×27×1152 → 14×14×4608
v_compressed = PixelShuffle_2x2(v)   // 196 tokens, dim=4608

// For global thumbnail tile:
//   Append <tile_newline> after each row → 14×15=210 tokens
// For local tiles grid:
//   Append <tile_newline> after final column to mark row endings

// Step 2: MLP Projection
for each visual token t_i in visual_sequence:
    h_i = MLP_2layer(t_i)   // 4608 → d_LLM (1280/2048/2560)

// Step 3: Combine with text tokens
full_sequence = [h_visual | <view_separator> | h_text]  // ready for LLM
```

VL Adaptor 变体：(a) 简单线性投影（LLaVA-1.5）；(b) Q-Former / Perceiver Resampler（BLIP-2, InstructBLIP, Qwen-VL）——使用可学习的 query tokens 对视觉特征做交叉注意力，输出固定数量的 tokens；(c) MLP 投影 + pixel shuffle（InternVL2, DeepSeek-VL2）；(d) MLP 投影 + convolution compression（Qwen2-VL）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
VL Adaptor 在训练的三阶段中扮演不同角色：(1) VL Alignment 阶段——LLM 冻结，仅训练 vision encoder + VL Adaptor，快速建立模态间的 embedding 空间对齐；(2) VL Pretraining 阶段——全参数训练，VL Adaptor 随整体模型一起调优；(3) SFT 阶段——继续全参数训练。这种 staged training 是 LLaVA-style VLM 训练的标准范式。训练 loss 仅计算在文本 token 上（包括 visual token 后的 answer token 和 special token），不计算在 visual token 上。

涉及论文标题：
- DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding

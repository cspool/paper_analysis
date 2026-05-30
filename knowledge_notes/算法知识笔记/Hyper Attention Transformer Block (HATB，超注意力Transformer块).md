## Hyper Attention Transformer Block (HATB，超注意力Transformer块)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hyper Attention Transformer Block (HATB) 是 mPLUG-Owl3 提出的轻量级多模态 Transformer 扩展模块。其核心创新在于：在语言模型的 Transformer block 中，将 cross-attention（文本查询→视觉特征）与 self-attention（文本内部）**并行执行**，而非 Flamingo 的串行插入方式。HATB 仅在 LLM 中稀疏替换少量层（如 Qwen2 28 层中的 4 层 [0, 9, 17, 25]），并通过四个关键设计实现高效多模态融合：(1) 共享 LayerNorm——复用 Transformer 原生 LN 同时对文本和视觉特征做归一化；(2) Modality-Specific KV Projection——视觉的 K/V 投影权重用 LLM 预训练 KV 权重初始化（W_img_KV ∈ R^{2D×D}）；(3) 共享 Query——cross-attention 的 Q 直接复用 self-attention 的 Q，使 LLM 的语言知识指导视觉特征选择；(4) Adaptive Gating——基于文本特征 Sigmoid 门控融合 self-attention 和 cross-attention 输出。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
HATB 在 LLM 推理中的计算流程：
```
# 输入: H_text ∈ R^{L×D} (文本隐状态), H_img ∈ R^{V×D} (视觉特征)
# HATB_layers = [l1, l2, ..., lK]  # 稀疏选择的层索引

for layer_idx in range(N_layers):
    # === 标准 Transformer block ===
    H_norm = LayerNorm(H_text)
    H_self_attn = SelfAttention(H_norm, causal_mask)
    H_text = H_text + H_self_attn

    if layer_idx in HATB_layers:
        # === Hyper Attention (与 self-attention 并行) ===
        # 1. 共享 LayerNorm: 复用同一 LN 对视觉特征归一化
        H_img_norm = LayerNorm(H_img)  # 与文本使用同一个 LN

        # 2. 获取 Query (复用 self-attention 的 Q)
        Q_text = W_Q(H_norm)  # 标准 Q 投影

        # 3. 视觉 KV 投影 (modality-specific，权重初始化自 LLM KV)
        K_img, V_img = split(W_img_KV(H_img_norm), dim=-1)

        # 4. MI-Rope 位置编码
        Q_rope = apply_rotary_pos(Q_text, pos_text)       # 文本位置
        K_img_rope = apply_rotary_pos(K_img, pos_images)  # 图像占位符位置

        # 5. Causal Cross-Attention
        A_cross = softmax(Q_rope @ K_img_rope^T / sqrt(d_k) + causal_mask_img)
        H_cross = A_cross @ V_img

        # 6. Adaptive Gating
        g = Sigmoid(W_gate^T @ H_text)           # g ∈ R^{L×1}, 逐 token
        H_fused = H_self_attn * g + H_cross * (1 - g)
        H_text = H_fused

    # === FFN ===
    H_text = H_text + FFN(LayerNorm(H_text))
```
关键点：cross-attention 在 self-attention 之后、FFN 之前执行；视觉特征不进入 LLM context window，序列长度始终为文本长度 L，不随图像数量增长；W_img_KV 在 Stage 1 仅训练此参数；4 层 HATB 达最佳效果，8 层反而退化；causal mask 确保自回归特性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
HATB 在 mPLUG-Owl3 中基于 Qwen2 实现。Vision Encoder: Siglip-400m（~400M 参数），Language Model: Qwen2（~7B），Linear Projection 对齐视觉隐空间到文本维度。HATB 额外参数量：W_img_KV ∈ R^{2D×D} 每层约 2D² 参数（D=3584），4 层共约 103M（占 LLM ~1.5%）；W_gate ∈ R^{D×1} 每层仅 D 参数。训练三阶段：Stage 1 仅训练新增模块 ~41M pairs；Stage 2 全参数训练多图数据；Stage 3 SFT。TP=4 单 GPU 显存 32-40GB。开源：https://github.com/X-PLUG/mPLUG-Owl。

涉及论文标题：
- mPLUG-Owl3__Towards_Long_Image-Sequence_Understanding_in_Multi-Modal_Large_Language_Models

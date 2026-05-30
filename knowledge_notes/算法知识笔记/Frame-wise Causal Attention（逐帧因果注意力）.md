## Frame-wise Causal Attention（逐帧因果注意力）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Frame-wise Causal Attention 是 StreamVGGT 将 VGGT 的全局 Alternating-Attention 改造为适合在线流式推理的注意力机制。核心改动：将 temporal attention 中的全局 attention（可访问所有帧的所有 token）替换为因果掩码的逐帧注意力——当前帧 t 的 Query 只能 attend 到帧 1..t 的 Key/Value，不能访问未来帧 t+1..T。

这与 LLM 的 autoregressive causal attention 哲学相似，但粒度不同：
- LLM：token-level causal（每个 token 只能看到前面的 tokens）
- StreamVGGT：frame-level causal（每帧可以看到当前及之前的帧，帧内 tokens 可以互见）

frame-wise 处理（而非 token-wise）的原因：视觉输入是逐帧到达的图像数据，每帧产生一批 patch tokens，帧内 spatial attention 无因果限制（帧内所有 patch 同时可得）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Frame-wise causal attention mask 结构
# 假设3帧，每帧2个token (R+N tokens 简化表示)
# Mask[t_i, t_j] = 0 if frame(i) ≤ frame(j) else -inf

# 序列: [f1_tok0, f1_tok1, f2_tok0, f2_tok1, f3_tok0, f3_tok1]
# Causal Mask (0=allow, -inf=block):
#   f1_tok0 f1_tok1 f2_tok0 f2_tok1 f3_tok0 f3_tok1
#   [0,      0,     -inf,   -inf,   -inf,   -inf  ]  # f1_tok0
#   [0,      0,     -inf,   -inf,   -inf,   -inf  ]  # f1_tok1
#   [0,      0,     0,      0,      -inf,   -inf  ]  # f2_tok0
#   [0,      0,     0,      0,      -inf,   -inf  ]  # f2_tok1
#   [0,      0,     0,      0,      0,      0     ]  # f3_tok0
#   [0,      0,     0,      0,      0,      0     ]  # f3_tok1

# 实现：按帧边界构建 block-wise causal mask
def frame_causal_mask(num_frames, tokens_per_frame):
    total = num_frames * tokens_per_frame
    mask = torch.zeros(total, total)
    for i in range(num_frames):
        for j in range(num_frames):
            if j > i:
                # 未来帧 → 该帧内所有 token 对后续帧不可见
                mask[i*tokens_per_frame:(i+1)*tokens_per_frame,
                     j*tokens_per_frame:(j+1)*tokens_per_frame] = float('-inf')
    return mask
```

与标准 causal attention 的关键区别：帧内 token 之间无因果限制（全互见），仅跨帧有因果限制。这使得 attention mask 呈 block-wise 结构，与 FlashAttention 的 block tiling 自然兼容。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
StreamVGGT 中，temporal causal attention 使用 FlashAttention 的 custom mask 参数：`flash_attn_func(q, k, v, causal=False)` 传入 custom attention mask。在 HuggingFace Transformers 框架下也可通过 `attention_mask` 参数实现。首帧 tokens 始终保留在 cache 中作为 "geometric reference"（XStreamVGGT 的设计选择），类似于 attention sink 但语义不同（编码场景的全局几何锚点而非注意力冗余接收者）。

涉及论文标题：
- XStreamVGGT__Extremely_Memory-Efficient_Streaming_Vision_Geometry_Grounded_Transformer_with_KV_Cache_Compression

## Hybrid Modality-Specific Sparse Attention / 混合模态特定稀疏注意力

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hybrid Modality-Specific Sparse Attention 是 VideoNSA 提出的注意力分配策略：在同一 LLM decoder 的每层中，按 token 的模态（vision vs text）分配不同的注意力机制。Vision tokens 使用 NSA（三支路 learnable 稀疏注意力），Text tokens 使用标准 GQA（dense attention）。关键 insight：(1) Vision tokens 高度冗余——帧间大量重复信息，适合 aggressive sparse attention；(2) Text tokens 承载精确语义指令，dense attention 确保指令跟随不退化；(3) 分离两路 attention 避免 vision 的稀疏化噪声污染 text reasoning。每层输出：o = [o_V; o_T]。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# VideoNSA Hybrid Attention (per layer)
X_V, X_T = split_by_modality(X, position_ids)
# Vision: NSA 3-branch
o_V = NSA_attention(X_V, block_size=64, n=32, w=256)
# Text: standard dense GQA
o_T = flash_attn(X_T, num_kv_heads=4)
# Merge
o = concat([o_V, o_T]); X_out = o + MLP(LayerNorm(o))
```

Annotations: 模态分离依据 position_ids。Vision block_size = 每帧 token 数（64 for Qwen2.5-VL），使每个压缩 block 对应完整一帧，归并时间冗余。28 layers 全部应用 hybrid attention（不加层选择）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现需修改 LLM decoder 的 attention forward：按 position_id 分离 token 序列 → vision path 调 NSA kernel → text path 调 FlashAttention kernel → 拼接。VideoNSA 基于 SWIFT + FLA 实现。这种 hybrid design 适用于任何视觉 token 冗余度高的多模态场景，对不同模态可选择不同 attention 策略（如音频用线性 attention，图像用 NSA）。

涉及论文标题：
- VideoNSA__Native_Sparse_Attention_Scales_Video_Understanding

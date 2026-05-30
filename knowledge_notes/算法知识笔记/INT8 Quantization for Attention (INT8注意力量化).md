## INT8 Quantization for Attention (INT8注意力量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
INT8 Quantization for Attention 是 SageAttention 提出的针对 Self-Attention 中 QK^⊤ Matmul 的 INT8 后训练量化。设计选择：(1) INT8 而非 FP8——consumer GPU (RTX4090/3090) 上 INT8 Matmul 是 FP16 的 4×、FP8 的 2×，且精度更高（INT8 QK cosine sim 99.54% vs E4M3 92.83% vs E5M2 77.95%）；(2) per-token 或 per-block 粒度——per-channel 不可行（dequant 需 outer axis scale）；(3) 1/√d 融合到 Q 量化中（on-chip fuse）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
δ_Q, Q̂_INT8 = per_block_quantize_int8(Q / sqrt(d))  # b_q=128
δ_K, K̂_INT8 = per_block_quantize_int8(K_smooth)      # b_kv=64
for Q̂_i, K̂_j:
    S_int32 = tl.dot(Q̂_i, K̂_j^T, input_precision='int8')  # Tensor Core u8·u8→s32
    S_ij = S_int32.to(tl.float16) * δ_Q[i] * δ_K[j]        # dequant
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
依赖 NVIDIA Tensor Core INT8 mma(u8.u8.s32)。Triton: `tl.dot(Q̂, K̂^T, input_precision='int8')`。per-block 量化: `scale=max(|x|)/127; x̂=clamp(round(x/scale),-127,127).to(tl.int8)`。Dequant: `S.to(tl.float16) * scales`。开源: https://github.com/thu-ml/SageAttention。

涉及论文标题：
- SageAttention2 Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization

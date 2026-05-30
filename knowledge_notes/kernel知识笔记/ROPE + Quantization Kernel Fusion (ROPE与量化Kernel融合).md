## ROPE + Quantization Kernel Fusion (ROPE与量化Kernel融合)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ROPE + Quantization Kernel Fusion 是 SageAttention 将 Rotary Position Embedding (RoPE) 操作与 Q/K 的 INT8 量化融合在同一个 GPU kernel 中完成的技术。目的：消除量化引入的额外 HBM I/O overhead。传统非融合方案：ROPE kernel: read Q,K from HBM → apply rotary → write Q_rope,K_rope to HBM；Quant kernel: read Q_rope,K_rope from HBM → quantize → write Q̂,K̂,δ_Q,δ_K to HBM。两次 HBM round-trip。融合方案：ROPE kernel 计算完 rotary 结果后，在 shared memory 中直接进行量化（on-chip），然后将 INT8 Q̂,K̂ 和 FP16 scales 写入 HBM，节省一次 round-trip。此外，SageAttention 还将 1/√d 系数融合到 Q 量化中（在 ROPE 后将 Q 乘以 1/√d，再量化），避免在 attention kernel 中额外做除法。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 非融合方案 (2 kernels, 2 HBM round-trips):
# Kernel 1 (ROPE):
Q_rope, K_rope = apply_rotary_pos_emb(Q, K, cos, sin)
write Q_rope, K_rope to HBM
# Kernel 2 (Quantize):
read Q_rope, K_rope from HBM
Q̂, δ_Q = per_block_quantize_int8(Q_rope * (1/sqrt(d)))
K̂, δ_K = per_block_quantize_int8(K_rope)
write Q̂, K̂, δ_Q, δ_K to HBM

# 融合方案 (1 kernel, 1 HBM round-trip):
# Kernel (Fused ROPE + Quant):
read Q, K from HBM
Q_rope, K_rope = apply_rotary_pos_emb(Q, K, cos, sin)  # register/SRAM
Q_scaled = Q_rope * (1/sqrt(d))                          # on-chip scale
Q̂, δ_Q = per_block_quantize_int8(Q_scaled)               # on-chip quant
K̂, δ_K = per_block_quantize_int8(K_rope)                 # on-chip quant
write Q̂, K̂, δ_Q, δ_K to HBM                              # single write
```
该融合减少了量化的 I/O overhead，使 quantize step 的额外 overhead 被 ROPE 的计算开销所 overlap。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Triton 实现：在 ROPE Triton kernel 的 epilogue 中插入量化逻辑。Triton 的 single-program 模型使得在同一 kernel 内混合 memory-bound (ROPE) 和 compute-bound (quantize) 操作变得自然。具体步骤：(1) ROPE 将 Q,K 从 HBM 加载到 SRAM；(2) 应用 rotary embedding（element-wise sin/cos 乘加）；(3) 在写入 HBM 前，在 SRAM 中计算 `max(|x_tile|)` 得到 scale，执行 `x̂ = round(clamp(x/scale, -127, 127)).to(tl.int8)`；(4) 将 INT8 数据和 FP16 scales 写入 HBM。该融合对 end-to-end 延迟有实质贡献——消除了量化 kernel 的单独 launch overhead 和一次 full tensor HBM round-trip。

涉及论文标题：
- SageAttention2 Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization

---

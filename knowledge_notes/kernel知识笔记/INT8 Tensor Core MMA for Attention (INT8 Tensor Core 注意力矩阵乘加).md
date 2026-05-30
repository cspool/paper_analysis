## INT8 Tensor Core MMA for Attention (INT8 Tensor Core 注意力矩阵乘加)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
INT8 Tensor Core MMA for Attention 是指在 Attention kernel 中利用 NVIDIA GPU Tensor Core 的 INT8 矩阵乘加指令（mma.u8.u8.s32）来加速 QK^⊤ Matmul。SageAttention 的选择理由：(1) INT8 throughput 在 consumer GPU (RTX4090) 上理论为 660 TOPS——是 FP16 (330 TFLOPS) 的 2×、FP8 (330 TFLOPS) 的 2×；(2) 实测 340 TOPS at headdim=64，达到理论峰值的 52%（FlashAttention2 仅 165 TOPS，50% FP16 峰值）；(3) SageAttention 在同一 kernel 内交替使用 INT8 (for QK^⊤) 和 FP16 (for PV) MMA 指令。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# SAGEAttn-B kernel (single Triton kernel launch)
# Q̂, K̂, V tiles 从 HBM 加载到 SRAM

# INT8 MMA for QK^⊤:
S_int32 = tl.dot(Q̂_i_INT8, K̂_j_INT8^T, input_precision='int8')  
# ↑ Triton → PTX: mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
# u8 inputs × 2 = 16-bit intermediates, accumulate to s32

# Dequantization (in FP16):
S_ij = S_int32.to(tl.float16) * δ_Q[i] * δ_K[j]
# ↑ per-block scale broadcast, light element-wise op

# Online Softmax + FP16 MMA for PV:
P̃_ij = exp(S_ij - m_new)  # FP16 exp
O += tl.dot(P̃_ij.to(tl.float16), V_j.to(tl.float16), out_dtype=tl.float16)
# ↑ FP16+FP16 accum MMA, f16.f16.f16
```
Kernel 配置（Table 12）: Q block b_q=128, KV block b_kv=64; Num Warps=4 (headdim=64) or 8 (headdim=128); Num Stages=3-5。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Triton 通过 `tl.dot()` 自动映射到 Tensor Core 指令。实现关键：(1) 输入张量必须是 int8 dtype (`tl.int8`)，值域 [-127, 127]；(2) Triton compiler 自动处理 Tensor Core tile 对齐（M=16, N=8, K=32 for INT8）；(3) 混合精度 kernel——在同一 Triton program 中先后调用 INT8 MMA 和 FP16 MMA——Triton 自动插入必要的 dtype conversion 指令；(4) 性能瓶颈分析：当 sequence length 较小（<512）时 attention 受 kernel launch overhead 主导，INT8 加速效果有限；当 sequence length 较大（>2048）时 compute-bound，INT8 MMA 加速效果显著。开源: https://github.com/thu-ml/SageAttention。

涉及论文标题：
- SageAttention2 Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization

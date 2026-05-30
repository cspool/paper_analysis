## Non-matmul FLOPs

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Non-matmul FLOPs（非矩阵乘浮点运算）指GPU kernel中不被Tensor Core（或等效矩阵乘专用硬件单元）加速的浮点运算。在NVIDIA GPU上，Tensor Cores专门加速矩阵乘加（MMA）操作（如FP16/BF16 matmul可达312 TFLOPs/s on A100），而elementwise操作（加法、乘法、指数、比较、reduction等）由CUDA Cores执行，吞吐远低于matmul。A100上FP32 non-matmul峰值仅19.5 TFLOPs/s，与matmul峰值312 TFLOPs/s之比为1:16——即每个non-matmul FLOP实质比matmul FLOP"贵"16倍。因此，要维持高总体吞吐（如>50%峰值），需尽量减少non-matmul FLOPs的比例，让GPU尽可能多地执行matmul操作。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FlashAttention v1 forward每次内迭代涉及的non-matmul操作及FlashAttention-2的优化：

```
# FlashAttention v1 每次内迭代的non-matmul FLOPs（per row block, B_r行）:
m_new = max(m_old, rowmax(S))            # B_r次compare (non-matmul)
m_rescale = exp(m_old - m_new)            # B_r次exp, B_r次sub (non-matmul)
ℓ_rescale = exp(m_old - m_new) * ℓ_old    # B_r次mul (non-matmul)
P = exp(S - m_new)                        # B_r×B_c次exp, 同量sub (non-matmul)
ℓ_new = ℓ_rescale + rowsum(P)             # B_r×B_c次add (reduction), B_r次add (non-matmul)
# FlashAttention v1额外: diag(ℓ)^{-1} rescale on O_old
O = diag(ℓ_new)^{-1} @ (diag(m_rescale) @ (diag(ℓ_old) @ O_old) + P @ V)
# 其中 ℓ_old/ℓ_new rescale: B_r×d次multiply (non-matmul)
# 以及 1/ℓ_new rescale on P: B_r×B_c次multiply (non-matmul)

# FlashAttention-2 每次内迭代（减少的non-matmul）:
# 1. 去掉O_old的ℓ-based rescale（省B_r×d次mul per iteration）
# 2. 去掉P的1/ℓ rescale（省B_r×B_c次mul per iteration）
# 仅保留:
m_rescale = exp(m_old - m_new)            # B_r次exp+sub
P_tilde = exp(S - m_new)                  # B_r×B_c次exp+sub
ℓ_new = m_rescale * ℓ_old + rowsum(P)     # B_r次mul + reduction
O_tilde = diag(m_rescale) @ O_tilde + P_tilde @ V  # B_r×d次mul (non-matmul)
# 最终一次性rescale: O = diag(ℓ)^{-1} @ O_tilde  # B_r×d次mul（仅1次/row block）
```

总计：FlashAttention-2 per iteration省去约`B_r×d + B_r×B_c`次non-matmul multiply，对典型block sizes (B_r=128, B_c=128, d=128)约省`128×128 + 128×128 = 32768`次non-matmul op/iteration。T_c=64时省约2M non-matmul ops/row block。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

减少non-matmul FLOPs的通用策略（从FlashAttention-2推广）：(1) 推迟归一化/缩放操作到所有累加完成后，而非每次迭代都做；(2) 使用fused multiply-add（FMA）将multiply和add合并为单一指令（但仍受CUDA core吞吐限制）；(3) 将scalar rescale因子（如diag(m_rescale)）与后续matmul合并——例如`O_tilde * m_rescale`可表达为`diag(m_rescale) @ O_tilde`，但无法完全纳入Tensor Core MMA；(4) 利用硬件特殊功能单元（MUFU.EX2 for exp2, MUFU.RSQ for 1/sqrt等）加速特定non-matmul操作。FlashAttention-2通过第(1)条策略将ℓ-based rescale从per-iteration推迟到final，显著减少non-matmul FLOPs。

涉及论文标题：
- FlashAttention-2 Faster Attention with Better Parallelism and Work Partitioning

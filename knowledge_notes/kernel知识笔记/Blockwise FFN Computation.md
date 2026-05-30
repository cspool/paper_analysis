## Blockwise FFN Computation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Blockwise FFN Computation 是 SRAMFFN kernel 中沿 d_ff 维度分块迭代计算的核心技术。与标准 FFN 的 monolithic 计算（先完整计算 gate/up 中间 tensor，再完整计算 down projection）不同，blockwise 策略将 FFN 参数矩阵 K/U/V 沿 d_ff 维度切分为 M 个 block，每个 block 单独完成 (SiLU(QK_m^T)⊙(QU_m^T))·V_m 计算并将结果累加到 output accumulator。关键数学等价性：(Σ_m SiLU(QK_m^T)⊙(QU_m^T))·V_m 正好等于对完整参数的 FFÑ 输出，因为 FNÑ 对 d_ff 维是线性的（通过累加）。FNÑ(Q;K,U,V) 在 d_ff 维度上的分离性质使得无需完整中间 tensor 即可累积出正确结果，这与 attention 中 online softmax 的 running statistics 维护性质本质相同。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Blockwise FFN 的正确性证明（为什么可以在不知道全局中间结果的情况下累积）:

# 原始计算（完整 materialize）:
# gate_full = SiLU(Q @ K_full.T)       # [L, d_ff]
# up_full   = Q @ U_full.T             # [L, d_ff]
# out = (gate_full * up_full) @ V_full # [L, d_h]

# 分块证明:
# 将 K/U/V 沿 d_ff 均分: K = [K_1; K_2; ...; K_M], 每块 b 行
# Q @ K.T = [Q@K_1.T, Q@K_2.T, ..., Q@K_M.T]  # concat along last dim
# 同理 Q @ U.T = [Q@U_1.T, ..., Q@U_M.T]
#
# SiLU(Q@K.T) * Q@U.T =
#   [SiLU(Q@K_1.T), ..., SiLU(Q@K_M.T)] * [Q@U_1.T, ..., Q@U_M.T]  # blocked concat
#   = [SiLU(Q@K_1.T)*Q@U_1.T, ..., SiLU(Q@K_M.T)*Q@U_M.T]
# 令 A_m = SiLU(Q@K_m.T) * Q@U_m.T ∈ R^{L×b}
#
# (SiLU(QK.T)*QU.T) @ V =
#   [A_1, A_2, ..., A_M] @ [V_1; V_2; ...; V_M]    # blocked matmul
#   = Σ_m A_m @ V_m                                   # accumulate!
#
# Q.E.D.: 分块的输出累加 = 全量的输出

# 每个 block m 的计算（forward fused kernel inner loop）:
for m in range(M):
    # Load {K_m, U_m, V_m} from HBM → SRAM (b × d_h each)
    # Compute A_m in SRAM: [BLOCK_SEQ, b]
    # Accumulate: O_acc += A_m @ V_m  → [BLOCK_SEQ, d_h]
    # A_m 用完即丢弃（不写回 HBM）

# 内存分析 (370M FlashMHF: d_h=128, d_e=342, BLOCK_INTER=64):
# 单 block SRAM footprint:
#   Q_blk:  BLOCK_SEQ × d_h × 2B            = 8K × 128 × 2  ≈ 2MB
#   K_tile: d_h × BLOCK_INTER × 2B          = 128 × 64 × 2  ≈ 16KB
#   U_tile: BLOCK_INTER × d_h × 2B          = 64 × 128 × 2  ≈ 16KB
#   V_tile: BLOCK_INTER × d_h × 2B          = 64 × 128 × 2  ≈ 16KB
#   O_acc:  BLOCK_SEQ × d_h × 2B            ≈ 2MB
#   A:      BLOCK_SEQ × BLOCK_INTER × 2B    = 8K × 64 × 2   ≈ 1MB
#   Total per SM: ≈ 5MB (可装进 H100 228KB SMEM with tiling)
#   实际需更细粒度 tiling: BLOCK_SEQ 可能需缩小至~1K
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Blockwise FFN Computation 的实现依赖于几个关键技术保障：
1. **Tile size selection**：沿 d_ff 的 BLOCK_INTER 应使得 K_tile·d_h·2B + U_tile·d_h·2B + V_tile·d_h·2B 较小（~50-100KB），而沿 sequence 的 BLOCK_SEQ 应尽可能大（提高 arithmetic intensity）但不超过 SMEM 限制。FlashMHF 论文未具体给出 BLOCK_SEQ/BLOCK_INTER 的数值。
2. **Backward pass 的 blockwise 策略**：backward 需要 saved Q 和 saved K/U/V（forward 中间结果），但不需要 saved A（完整中间激活）——因为 backward 可以 re-derive A_m 从 saved Q and K_m/U_m（recomputation 而非 storage）。这呼应了 FlashAttention backward 的 recompute 策略。
3. **Sub-network 循环嵌入**：blockwise 的内层循环（over d_ff blocks of size b）嵌套在外层循环（over E sub-networks）内。当 block boundary 与 sub-network boundary 对齐时（d_e mod BLOCK_INTER = 0），无需特殊处理。Router R 在 sub-network 切换时通过 producer prefecth 更新。
4. **与其他 blockwise 技术的对比**：FlashAttention 的 blockwise 沿 sequence dim（L）分块以消除 O(L²) 的 QK^T tensor；SRAMFFN 的 blockwise 沿 parameter dim（d_ff）分块以消除 O(L·d_ff) 的 intermediate activation。本质相同——将大中间 tensor 的计算拆成小块在 SRAM 中逐一处理。

涉及论文标题：
- Flash Multi-Head Feed-Forward Network

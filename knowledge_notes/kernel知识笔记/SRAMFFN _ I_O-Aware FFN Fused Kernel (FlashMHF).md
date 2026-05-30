## SRAMFFN / I/O-Aware FFN Fused Kernel (FlashMHF)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

SRAMFFN 是 FlashMHF 论文提出的 I/O-aware fused kernel，用于高效计算 Multi-Head FFN forward/backward pass。其设计理念直接类比 FlashAttention 的 I/O-aware online softmax：将大中间激活 tensor (SiLU(QK^T) ⊙ (QU^T)) ∈ R^{L×d_ff} 的计算沿 d_ff 维度分块（blockwise），每 block 的计算结果直接累加到 SRAM-resident output accumulator，避免在 HBM 中 materialize 完整的中间 tensor。核心公式：O ← 0; for m=1..M: O += (SiLU(Q·K_m^T) ⊙ (Q·U_m^T)) · V_m。其中 K_m, U_m, V_m ∈ R^{b×d_h} 是第 m 个参数的 block（b = BLOCK_INTER），M = d_ff / b。每 block 的中间结果 (SiLU(·)⊙(·)) ∈ R^{L×b} 在 SRAM 中计算后立即与 V_m 相乘并累加到 O_acc（也在 SRAM 中），从不写入 HBM。最终仅 output O ∈ R^{L×d_h} 写回 HBM。由于 FlashMHF 的 narrow head design（d_h 小，如 128），每 block 的 SRAM footprint 可完全装进 on-chip memory。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# === SRAMFFN Forward: Blockwise Fused Computation (Algorithm 1 简版) ===
# 输入: Q ∈ R^{L×d_h}, K/U/V ∈ R^{d_ff×d_h}, R ∈ R^{L×E} (router weights)
# 输出: O ∈ R^{L×d_h}
# 参数: BLOCK_SEQ (Q tiling), BLOCK_INTER (K/U/V tiling along d_ff)

def sramffn_forward(Q, K, U, V, R):
    # 三层并行: grid(seq_block, head, batch)
    # 每个 thread block 处理一段 sequence tokens + 一个 head
    
    for seq_start in range(0, L, BLOCK_SEQ):
        Q_blk = Q[seq_start:seq_start+BLOCK_SEQ, :]   # [BLOCK_SEQ, d_h]
        O_acc = zeros([BLOCK_SEQ, d_h])                # SRAM resident
        
        for e in range(E):                             # 遍历 sub-networks
            R_rows = R[seq_start:seq_start+BLOCK_SEQ, e]  # [BLOCK_SEQ]
            
            for m in range(0, d_e, BLOCK_INTER):        # 沿 d_e 分块
                # === Load K/U/V tiles from HBM to SRAM ===
                K_tile = K[e, m:m+BLOCK_INTER, :].T    # [d_h, BLOCK_INTER]
                U_tile = U[e, m:m+BLOCK_INTER, :]      # [BLOCK_INTER, d_h]
                V_tile = V[e, m:m+BLOCK_INTER, :]      # [BLOCK_INTER, d_h]
                
                # === Compute on SRAM (intermediate NEVER leaves SRAM) ===
                M = Q_blk @ K_tile                      # [BLOCK_SEQ, BLOCK_INTER]
                N = Q_blk @ U_tile                      # [BLOCK_SEQ, BLOCK_INTER]
                A = SiLU(M) * N                         # element-wise
                A = A * R_rows[:, None]                 # apply router weight
                
                # === Accumulate to output (SRAM-resident) ===
                O_acc += A @ V_tile                     # [BLOCK_SEQ, d_h]
        
        # === Write final output to HBM (only once per sequence block) ===
        O[seq_start:seq_start+BLOCK_SEQ, :] = O_acc
    
    return O
    # Total HBM traffic: Q/R/K/U/V loads + O write = O(d_model·L)
    # vs naive: add intermediate A ∈ R^{L×d_ff} write+read = O(d_ff·L)
    # Memory reduction: d_ff/d_model = (8/3)X for SwiGLU, larger for MH-FFN
```

```
# === SRAMFFN Hopper 实现: Warp-Group Specialization (Algorithm 4) ===
# 关键: 利用 Hopper 的 async TMA + warp-group concurrency

# 配置:
# CON_WARPGRPS ≥ 2 消费者 warpgroups
# PROD_WARPGRPS = 1 生产者 warpgroup
# NUM_STAGES ring buffer for K/U/V tiles

# Producer warpgroup (异步预取):
for inter_tile in range(NUM_STAGES, total_tiles):
    # Wait consumer finishes processing current stage
    wait_for_consumer_stage_release(current_stage)
    # Async prefetch next (K, U, V) tile via TMA
    tma_prefetch(K_tile[next], U_tile[next], V_tile[next], to=stage_buf[current_stage])
    # If new sub-network, also prefetch router R
    if is_new_subnet_boundary(inter_tile):
        tma_prefetch(R_rows, to=router_buf)

# Each Consumer warpgroup c (独立并行, 不同 x-block):
Q_blk = load_Q_for_my_xblock()       # 每个consumer负责不同的seq partition
O_acc = zeros([BLOCK_SEQ, d_h])      # accumulator in SRAM

for inter_tile in range(total_tiles):
    # Wait producer has data ready for this stage
    wait_stage_ready(current_stage)
    # Compute on loaded K/U/V (identical logic for all consumers)
    M = Q_blk @ K_tile.T; N = Q_blk @ U_tile.T
    S = SiLU(M) * N
    S = S * r  # apply router
    O_acc += S @ V_tile
    # Signal producer: this stage can be reused
    signal_stage_done(current_stage)

store_to_global(O_acc)  # write final output
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

SRAMFFN 的实现分两个版本：
1. **Triton 版本**（Algorithm 1-3）：适用于 consumer GPU（RTX3090）和开发环境。使用 Triton 的 grid/block 编程模型，forward 单 kernel，backward 分两个 kernel（DQ/DR kernel + DK/DU/DV kernel）。Grid 维度三维：(seq_block, head, batch)。优点是易编写、可移植；缺点是 Hopper 上新特性（TMA、warp-group specialization）无法直接利用，cuBLAS GEMM 已高度优化的 baseline 对比下 latency speedup 较小。
2. **ThunderKittens 版本**（Algorithm 4-5）：针对 Hopper 架构（H100）优化。利用 TMA 异步预取、warp-group specialization、stage ring buffer。一个 producer warpgroup 负责所有 consumer 的数据预取，多个 consumer warpgroups 并行处理不同的 sequence partition。这允许 TMA 的 memory latency 被 consumer 的 computation 完全隐藏。

实际使用：(1) FlashMHF module 的 forward/backward 中直接替换 tri-GEMM + elementwise op 调用为 SRAMFFN kernel；(2) kernel 内部自动处理 multi-head 的 batch/head 并行化——grid 中 y=H, z=B 两个维度覆盖所有 (batch, head) pair；(3) router weights R 作为额外输入（类似 attention 的 causal mask），在 sub-network boundary 处通过 producer 预取同步。限制：当前 d_h 需整除 BLOCK_INTER 等对齐约束，d_h=128 天然满足。

涉及论文标题：
- Flash Multi-Head Feed-Forward Network
- FlashAttention (Dao et al., 2022) — I/O-aware 设计范式的源头

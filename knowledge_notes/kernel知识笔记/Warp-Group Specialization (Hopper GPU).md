## Warp-Group Specialization (Hopper GPU)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Warp-Group Specialization 是 NVIDIA Hopper 架构 (SM90+) 引入的 CUDA 编程特性，允许单个 thread block 内的 warps 按功能分组为多个 warp group，每个 group 独立执行不同任务（producer/consumer pattern），并通过异步 barrier（mbarrier）进行 fine-grained 同步。Hopper SM 支持在一个 cycle 内调度不同 warp group 的指令——producer warp-group 发出 TMA（Tensor Memory Access）异步加载指令后立即 yield，consumer warp-group 在同一 cycle 执行计算指令。这使得 memory latency 可被完全隐藏。在 FlashMHF 的 ThunderKittens kernel 中：1 个 producer warpgroup 通过 TMA 异步预取 K/U/V tiles 到 stage buffer（ring buffer of NUM_STAGES），2+ 个 consumer warpgroups 各自绑定不同的 x-block（sequence partition），在各自的 stage 上并行执行 M/N/S/O 计算。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# FlashMHF Hopper Kernel 的 Warp-Group 分工 (Algorithm 4):
# CUDA Kernel Block: 包含 1 producer + CON_WARPGRPS consumer warpgroups

# === Producer (1 warpgroup, 4 warps): ===
# 职责: TMA async prefetch Q, R, K, U, V into SRAM stage buffers
# 同步: 通过 mbarrier 与每个 consumer 协调

stage_ready = [mbarrier_for_stage_0, ..., mbarrier_for_stage_N-1]
stage_done  = [mbarrier_for_stage_0, ..., mbarrier_for_stage_N-1]

def producer():
    # Warmup: prefetch Q for all consumers, first R, first N stages of K/U/V
    for c in range(CON_WARPGRPS):
        tma_prefetch(Q[c], to=q_buffer[c])
    tma_prefetch(R[subnet=0], to=r_buffer)
    for s in range(NUM_STAGES):
        tma_prefetch(K[s], U[s], V[s], to=stage_buffer[s])
    arrive(stage_ready[0..NUM_STAGES-1])   # signal ready
    
    # Main loop
    for tile_idx in range(NUM_STAGES, total_tiles):
        stage = tile_idx % NUM_STAGES
        wait(stage_done[stage])                 # consumer 处理完此 stage
        
        # Prefetch next tile
        if is_new_subnet(tile_idx):
            tma_prefetch(R[next_subnet], to=r_buffer)
        tma_prefetch(K_next, U_next, V_next, to=stage_buffer[stage])
        arrive(stage_ready[stage])              # signal consumer

# === Consumer c (独立 warpgroups, 2+ 个并行): ===
def consumer(c):
    Q_blk = q_buffer[c]  # 此 consumer 的固定 Q partition
    O_acc = 0
    
    for tile_idx in range(total_tiles):
        stage = tile_idx % NUM_STAGES
        wait(stage_ready[stage])                # wait producer
        
        if is_new_subnet(tile_idx):
            r = r_buffer                        # load router weights
        
        # Compute on current stage's K/U/V tiles:
        M = Q_blk @ K[stage].T                  # Tensor core MMA
        N = Q_blk @ U[stage].T                  # Tensor core MMA
        S = SiLU(M) * N * r                     # CUDA core element-wise
        O_acc += S @ V[stage]                   # Tensor core MMA
        
        arrive(stage_done[stage])               # signal producer

# Timeline (2 consumers, 3 stages, 4 tiles):
# Producer:  |P0 P1 P2|       |P3     |       |       |  (prefetch)
# Consumer0: |   |C0_t0|C0_t1 |C0_t2| |C0_t3| |       |
# Consumer1: |   |C1_t0|C1_t1 |C1_t2| |C1_t3| |       |
#                 ↑ overlap: producer P3 与 C0_t2+C1_t2 并发
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Warp-Group Specialization 在 FlashMHF 外的主要实现/应用：
1. **FlashAttention-3 (Hopper)**：使用 2 个 warpgroups——1 个负责 softmax（CUDA core），1 个负责 MMA（tensor core），通过 warpgroup-level 异步 barrier 实现 producer-consumer overlap。这是 FlashMHF 设计范式的直接来源。
2. **CUTLASS 3.x Hopper kernels**：支持 warp-group specialization 的 GEMM kernel，producer warpgroup 通过 TMA load global→shared，consumer warpgroups 执行 MMA 计算。
3. **ThunderKittens 框架**：提供 warp-group 级别抽象，允许用户用 C++ template 编写包含多个 warpgroup 的 kernel（而非手写 inline PTX）。
4. **约束和注意事项**：(a) warpgroups 必须是 4 warps 的整数倍（Hopper warp scheduler 粒度）；(b) 需要显式 mbarrier 管理同步（不能用 __syncthreads——它同步 block 内所有 warps）；(c) register/shared memory 在 warpgroups 间需显式分区（通过 cudaGetMemPool/cudaMallocAsync 或编译期静态划分）。

涉及论文标题：
- Flash Multi-Head Feed-Forward Network
- FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision (Shah et al., 2024)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Running Maximum-Based Block Skipping是BLASST在FlashAttention kernel中实现的动态block跳过机制。FlashAttention的online softmax在遍历KV block时需维护running maximum m_i^{(j)}（每行attention score的当前最大值），用于safe softmax的数值稳定。BLASST复用这一已有统计量：每个KV block计算完QK^T后求block local maximum m̃_i^{(j)}，与running maximum比较——若 m̃_i^{(j)} - m_i^{(j)} < ln(λ)，则该block中所有score的softmax后值均小于λ（因每个score ≤ block max，而exp分数的"有效尺度"由running max决定），对最终输出的贡献可忽略，因此跳过后续所有操作。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

BLASST skip decision的CUDA kernel级实现（伪代码，映射到warp/warpgroup）：

```
# Prefill kernel - 每个mainloop iteration中的skip check:
# Thread/warp布局: MMA warp做BMM1/BMM2, softmax warpgroup做softmax

# === Step 1: BMM1 (所有warp参与，tensor core) ===
S_block = wgmma(Q_tile, K_tile)        # QK^T, tensor core MMA

# === Step 2: Local max (softmax warpgroup, CUDA core) ===
m_local = warp_reduce_max(S_block)      # intra-warp shuffle reduction
# 每个thread lane持有S_block中部分elements

# === Step 3: Running max update ===
m_running = max(m_prev, m_local)        # 在线更新，本来就有

# === Step 4: Skip check (BLASST新增，3条指令) ===
pred = (m_local - m_running < ln_lambda)  # per-thread predicate
all_skip = __all_sync(pred)               # VOTE instruction - warp内统一
if (lane_id == 0):
    atomic_or(&shared_skip_flag, all_skip) # ATOMIC to shared memory
    # 一个warp的一票即可决定整block跳过

__syncthreads()                           # 等待所有warp投票

# === Step 5: Conditional execution ===
if (shared_skip_flag):
    continue                              # 跳过softmax和BMM2 - 直接进入下一轮
else:
    # ... 正常softmax: exp(S - m_running), rowsum, rescale
    # ... 正常BMM2: P_tilde × V (tensor core MMA)
```

Skip decision仅增加的指令：(1) predicate基于已有registers（m_local和m_running本来就计算好了），(2) VOTE指令（~1 cycle in warp），(3) ATOMIC to shared memory（~30 cycles with latency hiding）。在0% sparsity时overhead为0.96-1.00× baseline——证明这些指令被pipeline完全隐藏于BMM1的tensor core计算之后。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Skip check在prefill和decode kernel中的实现差异：(1) prefill kernel：softmax warpgroup执行skip check，因prefill compute-bound，跳过的是CUDA core的EXP计算和tensor core的BMM2 MMA，HBM load的V仍然照常prefetch（利用可预测的memory access pattern）；(2) decode kernel：改为batched load scheduling——连续执行多个K^TQ的BMM1，收集skip pattern后再批量加载需要的V blocks，因decode memory-bound，跳过V的HBM load是关键。BLASST已集成到TensorRT-LLM和FlashInfer。

涉及论文标题：
- BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding

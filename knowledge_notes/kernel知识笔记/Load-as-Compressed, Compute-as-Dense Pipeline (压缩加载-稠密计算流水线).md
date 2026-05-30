## Load-as-Compressed, Compute-as-Dense Pipeline (压缩加载-稠密计算流水线)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Load-as-Compressed, Compute-as-Dense 是一种 GPU kernel 设计范式，由 FlashLLM [Xia et al., VLDB 2023] 首次提出，用于加速非结构化稀疏矩阵的 GPU 计算。核心思想：稀疏矩阵以压缩格式从 GPU global memory (HBM) 加载到 SM 的 registers（减少 HBM 带宽消耗），在 shared memory 中解压为稠密 tile，再送入 Tensor Core 执行标准 dense GEMM（利用 Tensor Core 的高 FP16 throughput）。Mustafar 将该范式从 LLM weight projection 层适配到 KV cache attention 的 decode 阶段。

Pipeline 的三个阶段（以 SpMV 为例）：
1. **gmem2reg**：将 bitmap-compressed tile 从 HBM 加载到 SM registers（仅加载 bitmap + nonzeros，不及 dense 的 128B/tile 完整数据）
2. **extract**：根据 bitmap 将 nonzeros 放置到 shared memory 的正确 dense 位置，完成解压
3. **smem2tc**：从 shared memory 加载 dense tile 到 Tensor Core fragment，执行标准 MMA（matrix multiply-accumulate）

为什么对 decode attention 有效：Decode 阶段的 Q×K^T 和 AttnScore×V 是 batch of matrix-vector products (MVs)，在 GPU 上 severely memory-bound——计算量 O(Td) 远小于内存访问量 O(Td)（arithmetic intensity ~1 FLOP/byte）。Load-as-compressed 范式直接减少 HBM 数据搬移量，将 bottleneck 从 memory bandwidth 转移到 compute。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。

```
# Load-as-Compressed, Compute-as-Dense for KV Cache SpMV
# Mustafar custom CUDA kernel (decode stage)

Kernel: mustafar_spmv_attention(Q, K_bitmap_compressed, V_bitmap_compressed,
                                K_local_dense, V_local_dense)

# 常量: TILE_SIZE = 64, THREAD_TILES_PER_WARP = 2

# === Part 1: SpMV for compressed historical KV ===
for each warp (handles 64 tokens × 64-channel tile):
    # Stage: gmem2reg (Load as compressed)
    # 每个warp的32线程从HBM加载压缩数据到registers
    sync_warp()
    for thread in 0..31:
        for t in 0..THREAD_TILES_PER_WARP-1:
            tile_idx = thread * THREAD_TILES_PER_WARP + t
            # Load bitmap (8 bytes) + tile_offset (2 bytes) + nonzeros
            reg_bitmap = K_bitmap[tile_idx]
            reg_nonzeros = K_nonzeros[K_offset[tile_idx] : 
                                      K_offset[tile_idx] + popcount(reg_bitmap)]
    
    # Stage: extract (Decompress to shared memory)
    # 用bitmap将nonzeros散布到shared memory的dense tile
    sync_warp()
    K_smem[64][64] = {0}     # shared memory dense tile
    for thread in 0..31:
        for each nonzero in thread's nonzeros:
            # bitmap确定该nonzero在64x64 tile中的(row, col)位置
            (row, col) = decode_position(bitmap, nonzero_idx)
            K_smem[row][col] = nonzero_val
    
    # Stage: smem2tc (Compute as dense via Tensor Core)
    sync_warp()
    # Tensor Core mma: Q_fragment × K_smem^T
    S_partial = tc_mma(Q_reg, K_smem)   # [1×64] × [64×64] → [1×64]

# === Part 2: Dense MV for local window (W=32) ===
S_local = Q @ K_local^T       # [1×d] @ [d×32] → [1×32], standard cuBLAS

# === Part 3: Merge ===
S = concat(S_partial_all_warps, S_local)    # [1×T]
A = softmax(S / sqrt(d))
O = SpMV_on_V(A, V_bitmap_compressed) + A_local @ V_local
```

Pipeline 重叠优化（双缓冲）：Mustafar 支持在 gmem2reg 加载 tile i+1 的同时，extract+smem2tc 处理 tile i，利用 GPU warp scheduler 的异步特性实现内存加载与计算的重叠。在 FlashLLM 原始设计中这是 3 级 pipeline。

术语一般如何实现？如何使用？

实现要点：
1. **适用条件**：仅当计算是 memory-bound（低 arithmetic intensity）时才有效。Decode attention 满足；prefill attention (Q_len > 1) 是 compute-bound，使用 FlashAttention 而非 load-as-compressed。
2. **Tile 粒度**：1×64 thread-tile × 32 threads/warp = 64×64 warp-tile。Mustafar 中每 thread 解压 2 thread-tiles per stage，每 warp 处理 64×64 tile。
3. **Tensor Core 兼容**：解压后 dense tile 使用标准 FP16 Tensor Core MMA 指令（mma.sync.aligned.m16n8k16），无需特殊硬件支持。
4. **性能效果**：Mustafar 50% sparsity: SpMV 耗时 = 81.07% of cuBLAS (1.23× speedup)；70% sparsity: 61.87% of cuBLAS (1.62× speedup)。speedup 受 KV cache 压缩比决定——更高的稀疏度意味着更少的 HBM 数据搬移。
5. **Batch size 限制**：batch=1 时 SM underutilization（threadblock < SM count），需要 batch≥4 才能充分发挥 GPU 并行度。在 batch=8 (Llama-3) 时端到端 2.23× throughput。

涉及论文标题：
- Mustafar__Promoting_Unstructured_Sparsity_for_KV_Cache_Pruning_in_LLM_Inference

---

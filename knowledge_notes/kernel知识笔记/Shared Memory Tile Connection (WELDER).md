## Shared Memory Tile Connection (WELDER)

术语是什么？
Shared Memory Tile Connection是WELDER中tile-graph在GPU shared memory（L1）层的连接机制：通过SetConnect(edge, SharedMem)将两个相邻operator-tile通过shared memory中的reuse-tile直接连接，使第一个operator-tile的输出数据tile留在shared memory中，被第二个operator-tile直接消费，无需经过DRAM往返。这是WELDER fusion能力的关键——传统的Ansor/TVM仅支持register级的element-wise融合（如Conv+ReLU），而shared memory级的连接可以处理更复杂的operator组合（如Matmul+Softmax、Conv+Pool等reduction-based operator pair）。

从kernel调度角度拆解术语：
Shared Memory Tile Connection的kernel执行过程：

```
GPU Kernel: Fused Matmul + Softmax (shared memory level connection)

// 全局参数
grid_dim = (ceil(M/BM), ceil(N/BN))
block_dim = (128 threads)  // GCD of both operator tile thread counts

For each thread block (bm, bn):
  // Step 1: Load input tiles from DRAM → shared memory
  LoadTiles:
    A_tile[BM×BK] ← DRAM[A_addr + bm*BM×BK]  (coalesced 128B transactions)
    B_tile[BK×BN] ← DRAM[B_addr + bn*BN×BK]  (coalesced 128B transactions)
  
  // Step 2: Matmul operator-tile execution
  ComputeTile (Matmul):
    for kk in 0..BK step 16:  // TensorCore fragment size
      A_frag[16×16] ← shared_mem[A_tile[kk:kk+16]]  // ldmatrix
      B_frag[16×16] ← shared_mem[B_tile[kk:kk+16]]  // ldmatrix
      C_accum[16×16] += A_frag × B_frag  // mma.sync (TensorCore)
    
    // C_accum written to shared_mem[C_tile[BM×BN]]
    // ← KEY: C_tile stays in shared memory, NOT written to DRAM
  
  // Step 3: Inter-operator data reuse in shared memory
  // Softmax reads C_tile directly from shared memory
  __syncthreads()  // ensure Matmul writes visible
  
  // Step 4: Softmax operator-tile execution
  ComputeTile (Softmax):
    for row in range(BM):
      // All threads cooperatively process row
      row_data = shared_mem[C_tile[row, 0:BN]]
      max_val = warp_reduce_max(row_data)
      exp_vals = __expf(row_data - max_val)
      sum_exp = warp_reduce_sum(exp_vals)
      D_tile[row, 0:BN] = exp_vals / sum_exp
  
  // Step 5: Write final result to DRAM
  StoreTiles:
    DRAM[D_addr + ...] ← D_tile[BM×BN]  (coalesced 128B transactions)
```

Shared memory management:
```
// Buffer allocation (bestfit):
// - A_tile: BM×BK×4 bytes (FP32)
// - B_tile: BK×BN×4 bytes
// - C_tile/D_tile: BM×BN×4 bytes (reuse-tile, shared between Matmul output and Softmax input)
// - Padding: align to 32B for TensorCore, avoid bank conflicts

// Liveness analysis + bestfit offset calculation:
Allocation order: A_tile, B_tile, C_tile, (free A_tile after Matmul), ...
Finally: shared_mem[D_tile] → DRAM
```

术语一般如何实现？如何使用？
实现细节：(1) Load/Store Rewriting——TVM TIR pass将独立kernel的global memory access改写为shared memory access，添加memory fences防race condition；(2) Shared Memory Management——liveness analysis + bestfit算法统一管理所有shared memory buffer，考虑32B alignment；(3) Block/threadIdx remapping——Transpose等算子需blockIdx映射，2D thread block通过remapping与1D thread block并存；(4) Block size alignment——所有operator-tile的线程数取GCD作为统一block size（≥128 warp scheduler requirement且≤1024 max）。

涉及论文标题：
- Welder Scheduling Deep Learning Memory Access via Tile-graph

---

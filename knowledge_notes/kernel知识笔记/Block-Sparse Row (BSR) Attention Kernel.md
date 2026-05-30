## Block-Sparse Row (BSR) Attention Kernel

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Block-Sparse Row (BSR) Attention Kernel 是 FlashInfer 提出的基于 BSR 稀疏矩阵格式的 attention CUDA kernel。BSR (Block Compressed Sparse Row) 是一种硬件友好的稀疏矩阵存储格式，将非零元素组织为大小为 $(B_r, B_c)$ 的连续 dense block，而非 CSR (Compressed Sparse Row) 格式中的单个元素散布。在 FlashInfer 中，BSR 用作 KV-cache 的统一存储抽象：KV-cache pages (如 vLLM PagedAttention 或 SGLang RadixAttention) 被映射为 BSR 矩阵的 non-zero blocks，page table / radix tree 结构被映射为 BSR 的 indices arrays (`kv_indptr` 行指针 + `kv_indices` 列索引)。FlashInfer BSR attention kernel 支持任意 block sizes $(B_r, B_c)$，其中 $B_r$ 与 query tile size $T_q$ 对齐（控制 SMEM 中 KV tile 的复用粒度），$B_c$ 由 KV-cache 管理算法指定（如 page size = 1 for token-level management）。

BSR 相比 CSR 的优势：(1) 在 GPU 上 BSR 提升 register reuse efficiency——block 内元素在 shared memory 中 contiguous 排列，适合 tensor core MMA 指令的 dense 操作；(2) 可跳过整个零 block 减少计算；(3) 当 block size 对齐硬件 MMA 指令维度（如 NVIDIA mma 最小 16×16）时，可直接利用 dense tensor core 路径。FlashInfer 的 BSR attention 支持更小的 block size（如 (1,1) vector sparsity），通过先将分散的 global memory tiles gather 到 contiguous shared memory 再用 dense tensor core 处理——这基于 Chen et al. (2021) 和 Li et al. (2022) 的 vector-sparse 技术。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

BSR Attention Kernel 的 GPU 执行伪代码（以 FlashInfer decode kernel, H100, B_r=T_q=1, B_c=1 per-page sparsity 为例）：

```
// ===== 输入 =====
// Q: ragged tensor, shape [total_tokens, nheads, head_dim]
// KV_cache: BSR matrix, shape [num_blocks, B_r, nheads_kv, head_dim]
// kv_indptr: [num_rows + 1], row pointers for BSR
// kv_indices: [nnz], column indices of non-zero blocks

// ===== Kernel Launch =====
// Grid: num_CTAs (persistent, fixed for CUDAGraph)
// Each CTA: processes assigned work chunks from scheduler plan

// ===== Per-CTA Persistent Loop =====
for each (query_row, kv_chunk_start, kv_chunk_len) in CTA_work_queue:
    // Step 1: Load Q tile from ragged tensor
    q_tile = ldgsts_128B(Q[query_row : query_row + T_q])  // → SMEM
    __syncthreads()
    
    // Step 2: Initialize online softmax state
    O_acc = zeros(T_q, head_dim)
    l_acc = zeros(T_q, 1)
    m_acc = -inf * ones(T_q, 1)
    
    // Step 3: Iterate over KV chunks
    for kv_offset in range(kv_chunk_start, kv_chunk_start + kv_chunk_len, T_kv):
        // Step 3a: Load sparse KV tile from BSR
        // Compute global memory addresses from BSR metadata
        block_row = query_row / B_r
        nnz_start = kv_indptr[block_row]
        nnz_end = kv_indptr[block_row + 1]
        for j in range(nnz_start, nnz_end, num_blocks_per_tile):
            block_col = kv_indices[j]
            // cp.async gather: scattered GMEM → contiguous SMEM
            k_tile_smem = cp_async_ldgsts_128B(
                KV_cache[block_col : block_col + T_kv/pages_per_block])
            v_tile_smem = cp_async_ldgsts_128B(
                V_cache[block_col : block_col + T_kv/pages_per_block])
        cp_async_commit()
        cp_async_wait()
        __syncthreads()
        
        // Step 3b: S = Q × K^T (Tensor Core WGMMA)
        S = WGMMA(q_tile_smem, k_tile_smem)  // [T_q, T_kv]
        
        // Step 3c: Online softmax update
        m_new = rowmax(S, dim=1)  // CUDA core REDUX
        m_new = max(m_acc, m_new)
        P = exp(S - m_new)  // MUFU.EX2
        l_new = rowsum(P, dim=1)  // CUDA core REDUX
        // Rescale previous accumulator
        O_acc = O_acc * exp(m_acc - m_new)
        l_acc = l_acc * exp(m_acc - m_new) + l_new
        m_acc = m_new
        
        // Step 3d: O += P × V (Tensor Core WGMMA)
        O_acc += WGMMA(P, v_tile_smem)  // [T_q, head_dim]
    
    // Step 4: Write partial attention state
    // AttentionState = (O_acc / l_acc, log(l_acc) + m_acc)
    partial_O[chunk_idx] = O_acc / l_acc
    partial_LSE[chunk_idx] = log(l_acc) + m_acc

// ===== Contraction Kernel =====
// Merge all partial attention states using ⊕ operator
O_final = zeros(...)
LSE_final = -inf
for each (O_partial, LSE_partial) assigned to this CTA:
    O_final = (exp(LSE_final) * O_final + exp(LSE_partial) * O_partial) 
            / (exp(LSE_final) + exp(LSE_partial))
    LSE_final = log(exp(LSE_final) + exp(LSE_partial))
```

关键 BSR 特有步骤：
- **Global→Shared Memory Data Movement**：BSR indices arrays (`kv_indptr`, `kv_indices`) 计算 non-contiguous KV-cache 地址 → `cp.async` (LDGSTS, 128B width) 从分散的 HBM 地址 gather 到 contiguous SMEM → SMEM 中数据变为 dense tile → Tensor Core WGMMA/HMMA 处理。这与普通 dense attention 的 affine address transform 不同——BSR 的地址计算需要读取 sparse indices 间接寻址。Head dimension 保持 contiguous（size = d, 常见 128 或 256），维持 coalesced memory access。
- **TMA 使用限制**：H100 TMA (Tensor Memory Accelerator) 不支持 non-affine memory access patterns（即 BSR 的间接寻址），因此 FlashInfer 仅在 dense contiguous KV-cache 上使用 TMA，sparse BSR 路径回退为 Ampere-style `cp.async` LDGSTS。两种加载路径在 shared memory 之后汇合，后续 dense MMA 路径完全相同。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FlashInfer 的 BSR attention kernel 实现：
- 基于 CUDA/CUTLASS 模板，实现 FA2 算法（Turing/Ampere/Ada, sm75-sm89）和 FA3 算法（Hopper, sm90a）
- JIT 编译生成：attention variant specification (CUDA functors) + task info (BSR block sizes, tile sizes) → template population → PyTorch JIT compiler → 编译为 custom operator
- 集成入 vLLM、SGLang、MLC-Engine：上层框架的 page table / radix tree → 直接映射为 BSR `kv_indptr`/`kv_indices` → 传入 FlashInfer kernel，无需中间展平转换
- 支持任意 $(B_r, B_c)$：$B_r=1$ 对应 per-token page size（vector sparsity），大 $B_r$ 对应 batch-level grouping（配合 composable formats）
- GitHub: https://github.com/flashinfer-ai/flashinfer (Apache-2.0)

涉及论文标题：
- FlashInfer Efficient and Customizable Attention Engine for LLM Inference Serving

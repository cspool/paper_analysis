## Vector Sparsity / Fine-Grained Block Sparsity

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Vector Sparsity 是一种细粒度 block sparse 格式，block size 为 $(1, B_c)$ 或 $(B_r, 1)$，即 block 在某一个维度上 size=1（"vector"）。传统的 block-sparse 格式通常要求 block sizes 是 tensor core MMA 指令维度的倍数（如 NVIDIA mma 最小维度 16），导致 block sizes 至少为 (16, 16) 或更大——这对 fine-grained sparsity patterns（如 token-level KV-cache page sparsity in Quest，或 speculative decoding tree attention）不够灵活。

FlashInfer 支持 vector sparsity（$B_c=1$ for page-level KV-cache sparsity）的关键技术来自 Chen et al. (2021) 和 Li et al. (2022)：先通过 gather/scatter 将分散的 global memory elements 搬运到 contiguous shared memory，然后在 dense shared memory 数据上使用 tensor core 进行计算。核心 trade-off：接受 gather/scatter overhead（比 direct dense access 慢）来换取避免处理零 block 的节省（特别是在高稀疏度 scenarios）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Vector sparsity (B_c=1) 的 FlashInfer kernel 数据流动：

```
// ===== BSR with B_c=1 (per-page sparsity) =====
// 每个 KV-cache page = 1 token → 1 BSR block = (B_r, 1) × head_dim elements
// B_r = T_q (query tile size, e.g. 16 for FA2, 64 for FA3)
// B_c = 1 (single page per block)

// 传统 dense 方法：加载所有 pages（含大量不需要的）
// Vector-sparse 方法：仅加载 non-zero pages

// ===== Global → Shared Memory Data Movement =====
// 对于 B_c=1，每个 non-zero block 在 K dimension 上只有 1 page
// 需要 gather 多个 pages 到一个 SMEM tile 才能形成 dense tensor core 输入

__global__ void vector_sparse_attention(
    Q, KV_cache_pages, kv_indptr, kv_indices, ...
) {
    // Step 1: 确定哪些 pages 是非零的
    nnz_start = kv_indptr[block_row];
    nnz_end = kv_indptr[block_row + 1];
    num_pages = nnz_end - nnz_start;
    
    // Step 2: 将多个分散的 pages gather 到 contiguous SMEM
    // K tile: 需要 T_kv 个 pages 组成一个 dense tensor core tile
    for (tile_start = nnz_start; tile_start < nnz_end; tile_start += T_kv) {
        num_pages_in_tile = min(T_kv, nnz_end - tile_start);
        
        // Gather: LDGSTS from scattered HBM to contiguous SMEM
        for (p = 0; p < num_pages_in_tile; p++) {
            page_idx = kv_indices[tile_start + p];
            // 每个 page 在 HBM 中可能不相邻
            k_smem[p * head_dim : (p+1) * head_dim] = 
                cp_async_ldgsts(KV_cache_pages[page_idx]);  // [head_dim]
        }
        
        // Step 3: 在 SMEM 中的 dense tile 上使用 tensor core
        // 此时 K tile 在 SMEM 中是 contiguous dense 的
        S = WGMMA(q_smem, k_smem);  // [T_q, T_kv], tensor core
        // ... online softmax, PV ...
    }
}

// 对比：若使用 dense FA2 模板（假设不可跳过 pages）
// for all pages in range(max_pages):
//     load dense page (may be zero/irrelevant)
//     compute (waste on zeros)
```

Vector sparsity 的效率取决于 sparsity ratio 和 gather overhead 的 trade-off：
- 高 sparsity（大量 KV-cache pages 被 skip，如 Quest token importance sparsity > 90%）：vector sparsity 优势巨大——避免 ~90% 的 unnecessary page loads
- 低 sparsity（大部分 pages 都被使用）：gather overhead > skip savings，退化为 dense 更优
- FlashInfer 的 heuristic：对于已知 sparse patterns（如 Quest mask、tree attention mask），优先使用 vector-sparse kernel

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Vector sparsity 的实现：
- 基于 Chen et al. (2021) "Efficient tensor core-based GPU kernels for structured sparsity" 和 Magicube (Li et al., 2022) 的 vector-sparse GEMM 技术
- FlashInfer 将其扩展到 FlashAttention context：gather scatter KV-cache pages → dense MMA for QK^T and PV → online softmax
- 在 FlashInfer 中通过 BSR format 的参数化支持：任意 $(B_r, B_c)$ 值，$B_c=1$ 即 vector-sparse
- 关键 CUDA 实现：使用 `cp.async` (LDGSTS) 指令进行 gather——每个 page 一次 LDGSTS transaction (128B width) → commit group → wait → SMEM 中形成 dense tile
- TMA 不支持 vector sparsity（TMA 仅支持 affine/regular access patterns）→ vector-sparse kernel 回退 Ampere-style `cp.async` 路径

涉及论文标题：
- FlashInfer Efficient and Customizable Attention Engine for LLM Inference Serving

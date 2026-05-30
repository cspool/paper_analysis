## Bitmap-based Sparse Format (基于位图的稀疏格式)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Bitmap-based Sparse Format 是一种用于表示和存储非结构化稀疏矩阵的紧凑格式，由 Coruscant [Joo et al., MICRO 2025] 首次提出，旨在解决传统稀疏格式（CSR、COO）在 30%-70% 中等稀疏度下 metadata overhead 过高的问题。核心设计：将矩阵按 tile（1×64 的列向量）分块，每个 tile 用 64-bit bitmap 标记非零元素位置（1=nonzero, 0=zero），tile offset 寻址起始位置，仅存储非零元素值。

格式结构：每 tile 包含三个组件：
- **tile_offset** (uint16)：指向该 tile 在 compressed nonzeros 数组中的起始位置
- **bitmap** (uint64)：64-bit 掩码，bit[j]=1 表示该 tile 内第 j 个元素非零
- **nonzeros** (fp16 array)：紧凑存储的非零元素值，长度为 popcount(bitmap)

内存开销分析：
- Dense tile (64 fp16): 128 bytes
- 50% sparse tile: 32 fp16 values (64B) + bitmap (8B) + offset (2B) = 74B → 57.8% of dense
- 70% sparse tile: 19.2 fp16 values avg (38.4B) + 8B + 2B = 48.4B → 37.8% of dense
- 实际 Mustafar 50% 稀疏度达 65% 压缩比（含 padding 到 8 的倍数对齐 coalesced memory access），70% 稀疏度达 45%

在 Mustafar 中，该格式被扩展用于 KV cache 的压缩存储和直接计算：Key cache column-tiling 沿 token 维度，Value cache column-tiling 沿 channel 维度（因乘法方向不同），channel-major traversal 支持新 token 尾部追加。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。

```
# Bitmap Compressed KV Cache 格式定义（per tile）
struct CompressedTile:
    tile_offset: uint16     # 该tile非零元素在compressed buffer中的起始偏移
    bitmap: uint64          # 64-bit非零位置掩码
    # nonzeros存储在单独的contiguous buffer中
    # nonzeros[tile_offset : tile_offset + popcount(bitmap)]

# Compressed Key Cache 布局（column-tiling沿token维度）
# Key cache ∈ R^{T×d}, tiling: 每1×64 tile
# 共 T × ceil(d/64) 个tile

# SpMV over bitmap-compressed Key cache (Custom CUDA kernel)
# 输入: Q ∈ R^{1×d}, K_compressed (bitmap format)
# 输出: S ∈ R^{1×T} (attention scores)

Kernel: bitmap_batch_spmv(Q, K_compressed)
    # Grid: (num_heads, ceil(T/64))
    for each warp in grid:
        warp_tile = K_compressed[warp_id]  # 64个连续token的tile
        
        # Pipeline Stage 1: gmem2reg (Load compressed)
        for thread in warp:  # 32 threads
            # 每thread处理2个thread-tile (2×64 elements = 128)
            for t in 0..1:
                tile_idx = thread * 2 + t
                compressed = load_compressed_tile(tile_offset[tile_idx], bitmap[tile_idx])
                # compressed: (bitmap, nonzeros[]) 加载到register
                
        # Pipeline Stage 2: extract (Decompress to shared memory)
        for thread in warp:
            for element in nonzeros:
                # 用bitmap确定dense 64×64 tile中的正确位置
                smem[dense_row][dense_col] = element
                
        # Pipeline Stage 3: smem2tc (Tensor Core dense GEMM)
        S_partial = Q @ K_smem^T   # Tensor Core MMA, [1×d] @ [d×64] → [1×64]
        
    return concat all warp S_partial
```

术语一般如何实现？如何使用？

实现要点：
1. **压缩**：Triton kernel 实现 GPU 并行压缩——每个 tile 独立执行 mask→bitmap+nonzeros 的 pack 操作。
2. **Padding 对齐**：nonzeros 长度 padding 到 8 的倍数，确保 GPU memory coalesced access（每 128B cache line 对齐）。
3. **KV cache 追加**：channel-major traversal——新 token 的压缩 KV 直接追加到 buffer 末尾，无需重新组织已有数据。
4. **FlashAttention 兼容**：prefill 使用 FlashAttention（dense 计算），prefill 后压缩；decode 用 SpMV kernel 直接处理压缩格式。开源实现：https://github.com/dhjoo98/mustafar。

涉及论文标题：
- Mustafar__Promoting_Unstructured_Sparsity_for_KV_Cache_Pruning_in_LLM_Inference

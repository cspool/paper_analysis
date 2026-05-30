## Block Sparse Flash Decoding Kernel

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Block Sparse Flash Decoding Kernel 是 SeerAttention-R 提出的专门用于块稀疏注意力 decode 阶段的 GPU kernel。它扩展了 FlashAttention 的 flash decoding 模式，支持动态块稀疏性：kernel 接收 AttnGate 预测的 selected block indices，在遍历 KV cache 时只访问被选中的 blocks，跳过无效 entries。这消除了稀疏解码中不必要的 HBM 访存和计算，使实际加速接近理论值（speedup ≈ 1/(1-sparsity)）。

关键设计选择：
1. **3D Grid Launch**：沿 (batch, heads_kv, num_splits) 三维 launch，支持多 query group 和 KV shard 的并发计算
2. **按 max_selected_blocks 划分 split**（而非 total_blocks）：确保 sparsity 不均匀时 SM 间负载均衡
3. **wgmma 指令**（H100）：padding query head groups 到 64 以利用 warp group MMA 指令
4. **双重实现**：TileLang（自动应用 tiling/warp specialization/pipelining）和 Triton（相同调度策略的手动实现）两个版本

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。

```
// Block Sparse Flash Decoding Kernel (TileLang 伪代码)
// Grid: (batch, heads_kv, num_splits)，其中 num_splits = ceil(max_selected_blocks / BLOCKS_PER_SPLIT)

__global__ void block_sparse_flash_decode(
    float* Q,           // [batch, num_kv_heads, d_head] — 单 decode token
    float* K_cache,     // [seq_len, num_kv_heads, d_head]
    float* V_cache,     // [seq_len, num_kv_heads, d_head]
    int* block_indices, // [batch, num_kv_heads, max_selected_blocks]
    float* O,           // [batch, num_kv_heads, d_head]
    int max_selected_blocks, int block_size, float sm_scale
) {
    int bid = blockIdx.x, hid = blockIdx.y, sid = blockIdx.z;
    
    // 每个 split 处理 block_indices 的一个连续子集
    int blocks_per_split = ceil_div(max_selected_blocks, num_splits);
    int start_idx = sid * blocks_per_split;
    int end_idx = min(start_idx + blocks_per_split, max_selected_blocks);
    
    // 加载 Q 到 SRAM
    float Q_local[d_head] = load_tile(&Q[bid * num_kv_heads * d_head + hid * d_head]);
    
    // Online softmax 状态
    float O_local[d_head] = {0}, lse = -inf, m = -inf;
    
    for (int i = start_idx; i < end_idx; i++) {
        int block_id = block_indices[bid * num_kv_heads * max_selected_blocks 
                                      + hid * max_selected_blocks + i];
        if (block_id == INVALID) break;  // 提前终止
        
        // 仅加载被选中的 KV block
        float K_tile[block_size][d_head] = load_tile(
            &K_cache[block_id * block_size * num_kv_heads * d_head + hid * d_head]);
        float V_tile[block_size][d_head] = load_tile(
            &V_cache[block_id * block_size * num_kv_heads * d_head + hid * d_head]);
        
        // S = Q @ K^T * sm_scale (Tensor Core: wgmma on H100)
        float S_local[block_size];
        mma(S_local, Q_local, K_tile);  // [1, block_size]
        for (int j = 0; j < block_size; j++) S_local[j] *= sm_scale;
        
        // Online softmax update (FlashAttention 标准流程)
        float m_new = max(m, rowmax(S_local));
        float lse_new = m + log(exp(lse - m) + sum(exp(S_local - m)));
        float alpha = exp(m - m_new);
        float beta = 1.0 / exp(m_new - m);
        
        // O = alpha * O_prev + beta * exp(S - m_new) @ V
        for (int j = 0; j < d_head; j++) {
            O_local[j] *= alpha;
            float acc = 0;
            for (int k = 0; k < block_size; k++)
                acc += beta * exp(S_local[k] - m_new) * V_tile[k][j];
            O_local[j] += acc;
        }
        m = m_new; lse = lse_new;
    }
    
    // 写回 HBM（后续由 flash decoding 的 reduction kernel 合并各 split 的结果）
    store_tile(&O[bid * num_kv_heads * d_head + hid * d_head], O_local);
}
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

TileLang 实现：利用 TileLang DSL 编写高层 tile 操作，TileLang 编译器自动应用以下优化：(1) Tiling — 自动确定最优 tile size；(2) Warp specialization — 将线程分为 Producer（TMA 异步加载）和 Consumer（Tensor Core 计算）两组，通过 mbarrier 同步实现计算与访存 overlap；(3) Pipelining — 将循环展开为 prologue/steady/epilogue 三段，使下一块的数据加载与当前块的计算并行；(4) Tensorization + rasterization + swizzling — 优化 HBM 访存模式和 bank conflict。Triton 实现：手动编写相同调度策略的 kernel，提供对比 baseline。

性能特点：由于 decode kernel 为 I/O-bound，加速效果在长序列和大 batch 时最显著。bs=16, seqlen=128k, 90% sparsity 时 TileLang kernel 达 8.6× vs FA3 (理论 10×)。TileLang 实现比 Triton 实现快 1.7×。

涉及论文标题：
- SeerAttention-R: Sparse Attention Adaptation for Long Reasoning

---

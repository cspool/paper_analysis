## Atomic-Free GPU Parallel Dispatch Data Structure Construction

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Atomic-Free GPU Parallel Dispatch Data Structure Construction 是 MoEBlaze 提出的用于替代传统 sort-based MoE token dispatch 的 GPU kernel 设计方法。传统方法（如 MegaBlocks）通过 multi-pass radix sort 将 (expert_id, token_id) 对按键排序来分组 token，但 sort 需要多次 global memory pass（pass 数与 key width 成正比）和复杂的 multi-kernel pipeline（radix sort → segmented scan → index recovery），导致 kernel launch latency 高且 GPU 资源利用率低。MoEBlaze 用 3 步 atomic-free 并行构建流程替代 sort：每一步都设计为无原子操作的 GPU 并行执行，利用 shared memory prefix sum 和 warp-level reduction 避免全局内存冲突。

从kernel调度角度拆解术语：
```
// Step 1: Build Dense Token-Expert Map
// Grid: L×E threads or warp-mapped CTA grid
// 每个 warp 分配不相交的 token rows, 写入 dense_token_map
Kernel BuildDenseMap(topk_experts[L, K], dense_token_map[L, E]):
    token_id = blockIdx.x * blockDim.x + threadIdx.x
    if token_id >= L: return
    for slot in 0..K-1:
        expert_id = topk_experts[token_id, slot]
        dense_token_map[token_id, expert_id] = token_id  // 无 collision
    // 每个 (token, expert) pair 最多 write 一次

// Step 2: Compute Expert Lengths via Warp-Level Reduction
// Grid: E CTAs, 每个 CTA 处理一个 expert 列
Kernel ComputeExpertLengths(dense_token_map[L, E], expert_lengths[E]):
    expert_id = blockIdx.x
    local_count = 0
    for row in thread_range(0, L, blockDim.x):
        if dense_token_map[row, expert_id] != EMPTY:
            local_count += 1
    // Warp-level reduction within CTA
    expert_lengths[expert_id] = warpReduceSum(local_count)

// Step 3: Route Indices to Gates (Atomic-Free via Location Map)
// 两阶段: (a) tile-level scan in shared memory
//         (b) global offset addition
Kernel RouteIndices(dense_token_map[L, E], expert_offsets[E+1],
                    expert_token_indices[L*K]):
    expert_id = blockIdx.x
    tid = threadIdx.x
    // 3a. Tile-level exclusive scan in shared memory
    shared tile_counts[BLOCK_SIZE]
    // Count non-empty entries per tile
    count = (dense_token_map[row, expert_id] != EMPTY) ? 1 : 0
    tile_counts[tid] = count; __syncthreads()
    // Exclusive scan within tile (prefix sum)
    pos = exclusiveScan(tile_counts, tid)
    // 3b. Write to final position = expert_offset + tile_offset + pos
    if count > 0:
        expert_token_indices[expert_offsets[expert_id] + pos] =
            dense_token_map[row, expert_id]
```
关键设计：三步均无原子操作。Step 1 利用 expert ID per token 唯一性保证无 intra-warp collision。Step 2 的 warp-level reduction 在 shared memory 内完成。Step 3 的 tile-level scan 在 shared memory 内做 prefix sum，然后加全局 expert_offsets 得到 final position ID——location map 构建是确定性的，写入位置无冲突。

术语一般如何实现？如何使用？
- 输入：topk_experts [L×K]（int32 expert IDs，由 Gating Network 产生）
- 输出：expert_token_indices [L×K], expert_token_offsets [E+1], token_expert_indices [L×K], token_index_map [L×K]
- 与 sort-based dispatch 对比：sort ≈ 4+ global memory passes（radix sort）× O(LK)；MoEBlaze ≈ 3 passes（dense map fill + expert count + location write），每次 pass 的 global memory traffic 更少
- 在 H100 上衡量：dispatch kernel 成为训练的关键路径优化——与传统 sort dispatch 相比，kernel launch chain 从 6-8 降至 3，GPU utilization 更高
- 论文未明确说明开源链接

涉及论文标题：
- MoEBlaze: Breaking the Memory Wall for Efficient MoE Training on Modern GPUs

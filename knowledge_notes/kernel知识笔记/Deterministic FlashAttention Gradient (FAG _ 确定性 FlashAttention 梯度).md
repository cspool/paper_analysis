## Deterministic FlashAttention Gradient (FAG / 确定性 FlashAttention 梯度)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Deterministic FlashAttention Gradient (FAG) 是 LongCat-Flash 训练基础设施中实现的确定性后向传播 kernel。默认 FlashAttention 的 backward pass 使用 atomicAdd 对 dQ/dK/dV 沿不同维度进行归约——原子操作不保证执行顺序，导致同一输入在不同 run 间产生 bitwise 不同的梯度。这种非确定性使得：(1) 训练无法精确复现，(2) SDC (Silent Data Corruption) 检测困难（缺少 bitwise 一致的 baseline）。

LongCat-Flash 的 deterministic FAG 方案：使用有限 extra workspace 按确定性顺序累积各 tile 的部分梯度，替代默认的 atomicAdd unordered reduction。通过 double-buffer pipelining、tuned tiling schedules 和 load balancing 三项协同优化性能。结果：达到原始确定性版本的 1.6x 速度，非确定性版本的 0.95x。

LongCat-Flash 是首批在整个训练 pipeline 中实现端到端确定性的 LLM training system——包括 computation 和 communication 两部分。确定性确保任意 training step 可被多次重跑并产生 bitwise identical loss，使 SDC 检测成为可能。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Deterministic FAG Kernel (简化伪代码)

输入: Q, K, V, dO (BF16 tensors), 所有 shape 已知

// 将 dQ/dK/dV 计算划分为 tiles
tiles = partition(Q.shape, tile_size_per_SM)

// 每个 SM 分配 tiles (load balancing)
for sm_id, tile_set in balanced_split(tiles, num_SMs):
    sm_workspace = alloc_workspace()  // 确定性累加用 workspace

    // Double-buffer pipelining: 两组 buffer 交替使用
    buf_a = alloc_buffer()
    buf_b = alloc_buffer()

    for i, tile in enumerate(tile_set):
        cur_buf = buf_a if i % 2 == 0 else buf_b
        prev_buf = buf_b if i % 2 == 0 else buf_a

        // 异步加载当前 tile 数据 (TMA/async copy)
        load_async(tile.Q, tile.K, tile.V, tile.dO)

        // 如果上一 tile 完成, 将结果写入确定性累积的 workspace
        if i > 0:
            deterministic_accumulate(sm_workspace, prev_buf.dQ)
            deterministic_accumulate(sm_workspace, prev_buf.dK)
            deterministic_accumulate(sm_workspace, prev_buf.dV)

        // 计算当前 tile
        cur_buf.dQ, cur_buf.dK, cur_buf.dV = flash_attention_backward_tile(
            tile.Q, tile.K, tile.V, tile.dO
        )

    // 最后 tile 的归约
    deterministic_accumulate(sm_workspace, cur_buf.dQ)
    deterministic_accumulate(sm_workspace, cur_buf.dK)
    deterministic_accumulate(sm_workspace, cur_buf.dV)

// SM 间合并: 按确定性顺序 (如 SM ID 升序)
for sm_id in sorted(range(num_SMs)):
    merge_into_global_output(global_dQ, sm_workspaces[sm_id].dQ)
    ...
```

关键优化：
- **Deterministic accumulation**: 按 tile 顺序依次累加到 workspace，替代 atomicAdd
- **Double-buffer pipelining**: 当前 tile 计算 + 上一 tile 结果写回重叠执行
- **Tuned tiling**: 按 H800 SM 数量和 shared memory 大小优化 tile 尺寸
- **Load balancing**: 在各 SM 间均匀分配 tile 计算量

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：
1. Workspace 开销：需要存储各 tile 的部分梯度，开销约 O(num_tiles × tile_output_size)，在 H800 80GB 上可接受。
2. 性能 tradeoff：确定性 vs 速度。LongCat-Flash 的 0.95x non-deterministic 性能水平是 SOTA——此前确定性实现通常有 1.5x-2x 减速。
3. 与 SDC 检测的集成：FAG 是最敏感的 SDC 检测点（同时混合 tensor 和 vector 计算），通过 on-chip in-place recomputation 对比 bitwise 结果检测 SDC。
4. 应用于训练 full pipeline：FAG + Deterministic ScatterAdd + 确定性通信（pipelined all-gather/reduce-scatter 代替 all-to-all）共同实现端到端确定性。

涉及论文标题：
- LongCat-Flash Technical Report

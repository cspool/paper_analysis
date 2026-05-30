## Batched Load Scheduling for Memory-Bound Decode Kernel

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Batched Load Scheduling是BLASST为decode kernel设计的load调度优化，解决naive sequential pipeline中V load与skip check之间的scoreboard dependency导致的pipeline bubble问题。在decode阶段（query length=1），attention是memory-bound的，瓶颈在于HBM中KV cache的加载。BLASST的目标是根据skip check结果仅加载需要的V blocks来减少HBM traffic，但naive实现中：必须先完成BMM1(QK^T) → 才能做skip check → 才能决定加载哪些V → 导致连续的K load和V load之间出现依赖stall（pipeline bubble）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Batched Load Scheduling的执行流程：

```
# Decode kernel with Batched Load Scheduling
# 预取Batch大小: B (如B=4)

# === Phase 1: Back-to-back BMM1s ===
for b in 0..B-1:
    S_b = Q × K_{j+b}^T          # BMM1 batch: 连续发射QK^T
    store S_b to smem_buf[b]     # 存入共享内存buffer（query len=1, 很小）

# === Phase 2: Batch skip check ===
skip_mask = 0
for b in 0..B-1:
    m_local = max(S_b)           # 单query token, max即scalar
    m_running = max(m_running, m_local)
    if m_local - m_running < ln(λ):
        skip_mask |= (1 << b)     # 标记跳过

# === Phase 3: Conditional V loads ===
for b in 0..B-1:
    if (skip_mask & (1 << b)) == 0:
        V_{j+b} = load_from_hbm(KV_cache_addr + (j+b)*V_stride)  # 仅加载需要的V
        O = O * rescale + softmax(S_b) × V_{j+b}                  # BMM2
    # else: 跳过V load和BMM2，省HBM bandwidth

# Tradeoff: 需要B份S_b的shared memory buffer
# 但query len=1, B_r=1, 每个S_b仅B_c × sizeof(fp16) ≈ 128B
```

对比FlashAttention decode的sequential pipeline：V_j load → BMM1(QK_1^T) → BMM2 → V_{j+1} load → ...，每个iteration中V load必须在BMM1完成后才能发起（需V地址）。BLASST的batched方案：先批量做B个BMM1积累skip knowledge，再仅发射通过检查的V loads，消除了scoreboard stall。Figure 4a vs 4b: 38 vs 31 time units完成全部V loads。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Batched Load Scheduling在BLASST的decode kernel（target sm90a/sm100）中实现。适用条件：decode phase的query length=1（或很小），使多个S_b buffer的shared memory开销可忽略（128 bytes × B / buffer）。当query length较大时（如prefill），S_block的shared memory开销增大，batched策略不再适用。对于compute-bound的MLA decode，BLASST额外跳过softmax计算（不仅仅是V load），提供进一步speedup。

涉及论文标题：
- BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding

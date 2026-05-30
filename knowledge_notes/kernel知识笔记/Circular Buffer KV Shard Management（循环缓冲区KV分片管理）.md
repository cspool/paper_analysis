## Circular Buffer KV Shard Management（循环缓冲区KV分片管理）

术语是什么？
Circular Buffer KV Shard Management 是 PiKV 提出的 Expert-Sharded KV Storage 中的 per-shard 内存管理机制。每个 GPU shard 维护固定容量 S 的 circular buffer（循环缓冲区），通过 hash 函数 s(t,e) = (t mod N_tok) ⊕ (e mod N_exp) 将 KV 条目映射到唯一的 (shard, position)，实现 O(1) 时间插入和 O(1) 时间查找。Circular buffer 的固定容量保证无内存碎片和 reallocation，代价是 buffer 满时需驱逐最旧条目（FIFO 语义）。Buffer 大小 S 可根据 PiKV 的 closed-form 优化公式选择：S* = √(L/(KG))，在 sharding granularity 与 reuse coverage 之间求最优平衡。

从kernel调度角度拆解术语：
Circular Buffer KV Shard 的 kernel 级操作：

```
# === Data Structures (per GPU shard) ===
struct KVShard {
    // Circular buffer: fixed-size array + head pointer
    KeyBuffer   K_buf[S][d'];    // S slots, d' compressed dim
    ValueBuffer V_buf[S][d'];    
    Metadata    meta[S];          // {token_id, expert_id, timestamp, ...}
    uint32_t    head;             // next write position (0 ≤ head < S)
    uint32_t    count;            // current valid entries
};

# === O(1) Insert Kernel ===
def kv_shard_insert(shard, K_compressed, V_compressed, token_id, expert):
    pos = shard.head
    # Overwrite oldest entry (FIFO eviction)
    shard.K_buf[pos] = K_compressed        # O(d') memcpy
    shard.V_buf[pos] = V_compressed
    shard.meta[pos] = {token_id, expert, now()}
    shard.head = (pos + 1) % S              # advance head
    shard.count = min(S, shard.count + 1)

# === O(1) Lookup (by position) ===
def kv_shard_lookup(shard, token_id):
    # Hash token_id to buffer position
    pos = token_id % S
    # Verify metadata match (handle hash collisions)
    if shard.meta[pos].token_id == token_id:
        return (shard.K_buf[pos], shard.V_buf[pos])
    else:
        return CACHE_MISS

# === O(S) Scan for Query-Aware Selection ===
def kv_shard_select_pages(shard, q_t, threshold):
    selected = []
    for pos in range(shard.count):
        # Compute utility score for each cached entry
        u = compute_utility(
            shard.K_buf[pos], shard.V_buf[pos],
            shard.meta[pos],
            q_t
        )
        if u >= threshold:
            selected.append((pos, u, shard.meta[pos]))
    # Return top-K by utility
    return sorted(selected, key=lambda x: x[1], reverse=True)[:K]
```

**Buffer 大小优化**：
$$S^* = \sqrt{\frac{L}{KG}}$$
- L 过大 → S* 增大（更多 slot 支持更大 token 容量）
- K 增大 → S* 减小（调度器保留更多 page，减少 per-shard buffer 需求）
- G 增大 → S* 减小（更多 GPU 分担，per-GPU buffer 可缩小）

**Per-GPU Memory Formula**：
$$\mathcal{M}_{\text{total}} = \frac{2d}{\rho}\left(\frac{L}{GS} + KS\right)$$

术语一般如何实现？如何使用？
- PiKV CUDA 实现：`core/cuda/pikv_cuda.py` 中的 `moe_routing` 和 `top_k_experts` CUDA kernel 负责 KV tensor 的 gather/scatter 操作，circular buffer 逻辑在 `core/single/kvcache_centric_system.py` 的 PagedKVCache 中由 Python 管理。
- 与 vLLM block table 的关系：vLLM 的 physical block table 是动态分配的（需要 free list + allocation），PiKV 的 circular buffer 是固定大小（无 allocation 开销），适合 MoE 的 per-expert per-shard 细粒度 KV 存储。
- CUDA kernel 加速：`pikv_cuda.moe_routing(tokens, expert_weights)` → GPU parallel gather of KV from shards, `pikv_cuda.top_k_experts(scores, k)` → warp-level top-k on routing scores。
- 适用场景：MoE 推理的 per-shard KV 存储（S 典型值 256-512），要求低延迟 O(1) 操作的 streaming workload。不适合需要复杂驱逐策略的 workload（此时用 PiKV Scheduling 的多特征评分选择 active pages）。

涉及论文标题：
- PiKV KV Cache Management System for Mixture of Experts

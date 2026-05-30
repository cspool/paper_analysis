## Expert-Sharded Distributed KV Cache（专家分片分布式KV缓存）

术语是什么？
Expert-Sharded Distributed KV Cache 是 PiKV 提出的面向 MoE 架构的 KV cache 分布式存储策略。核心思想：将 KV cache 按 token 和 expert 两个维度 hash 分片到 G 个 GPU，每个 GPU 仅存储属于其分片的 token 的 KV，避免传统方案中每个 GPU 复制完整 KV 导致 O(EL) 的内存开销。分片函数 s(t,e) = (t mod N_tok) ⊕ (e mod N_exp) 确保每个 expert-token 对唯一映射到一个 GPU shard。每个 shard 维护 circular buffer（容量 S），O(1) 时间插入。Per-GPU 内存：M_kv = (2d/ρ)(L/(GS) + KS)，其中 L 为每 expert 全局 token 数，K 为 scheduler 保留的 page 数，S 为 buffer 大小。最优 buffer 大小 S* = √(L/(KG)) 可通过求导得出，对应最小内存 M* = (4d/ρ)√(KL/G)。

从系统架构角度拆解术语：
Expert-Sharded Storage 在多 GPU 推理中的运转流程：

```
# === Initialization ===
# G GPUs, E experts, N_tok tokens per shard
for gpu in range(G):
    # Each GPU initializes circular buffers for its assigned shards
    for (t_mod, e_mod) in assigned_shards[gpu]:
        C[gpu][e_mod][t_mod] = CircularBuffer(capacity=S)

# === KV Insert (per generated token) ===
def insert_kv(K_t, V_t, token_id, routed_experts):
    for e in routed_experts:
        s = hash_shard(token_id, e)
        # s = (token_id % N_tok) XOR (e % N_exp)
        target_gpu = s % G
        if target_gpu == local_gpu:
            C[e][s].append((compress(K_t), compress(V_t)))
        else:
            # RDMA transfer to target GPU (DistributedKVCachePool)
            rdma_send(target_gpu, (e, s, K_t, V_t))

# === KV Fetch (per decode query q_t) ===
def fetch_kv(q_t, routed_experts):
    pages = []
    for e in routed_experts:
        for s in relevant_shards[e]:
            if s maps to local_gpu:
                pages += C[e][s].get_active_pages(q_t, scheduler)
            else:
                pages += rdma_recv(remote_gpu(s), (e, s, q_t))
    return pages
```

关键设计决策：
1. **Hash 分片 vs 动态放置**：hash 提供确定性映射（O(1) 定位），无需中心化 metadata server；代价是负载可能不均（某些 expert 的 KV 访问频率更高）。
2. **Circular buffer vs 动态分配**：circular buffer 避免内存碎片和 reallocation；代价是 buffer 满时需驱逐旧条目。
3. **Per-expert 隔离 vs 全局 pool**：per-expert 隔离防止热门 expert 的 KV 挤占冷门 expert 的缓存空间，保证各 expert 有最小缓存保障。

术语一般如何实现？如何使用？
- PiKV 实现：`core/single/kvcache_centric_system.py` 中的 PagedKVCache（GPU/CPU/SSD 三级）+ DistributedKVCachePool（RDMA 跨节点）。
- 与 vLLM PagedAttention 的关系：PiKV 在 vLLM 的 block-level paging 之上增加 expert 维度分片。vLLM 的 physical block table 按 (token_position, block_id) 索引，PiKV 扩展为 (token_position, block_id, expert_id)。
- 适用场景：多 GPU/多节点 MoE 推理（E≥8, G≥4, L>32K），当单 GPU 无法容纳全量 KV cache 时。也适用于 expert parallelism 训练中的 KV 管理。
- 相关方法：RingAttention（sequence 维度分片，无 expert 维度）、DeepSpeed-MoE（expert parallelism for weights, not KV）、vLLM PagedAttention（无 expert sharding）。

涉及论文标题：
- PiKV KV Cache Management System for Mixture of Experts

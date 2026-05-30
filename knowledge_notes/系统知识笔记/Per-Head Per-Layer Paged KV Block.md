## Per-Head Per-Layer Paged KV Block

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Per-Head Per-Layer Paged KV Block 是 KV-Compress 对 vLLM PagedAttention 的 block 布局重构。原 vLLM PagedAttention 中，每个物理 cache block 存储某个请求在所有 transformer layers × 所有 attention heads 上的 b 个连续 token 的 KVs（即一个 block 包含 l×H×b 个 key/value 向量）。所有 layer 和 head 共享同一套 block table 索引——通过一个 block index 即可定位某 token 在所有 layers 和 heads 上的 KVs。

KV-Compress 将 block 重构为 per-head per-layer 粒度：每个物理 block 仅存储单个 KV head 的 b 个 token 的 KVs（即一个 block 仅包含 b 个 d 维向量）。Block table 从 B × L_max/b 扩展为 B × l × H × L_max/b，每个 (layer, head) 对有独立的 block table。

物理 cache 从 l 个 per-layer tensor K^(m) ∈ R^{N×H×b×d} 改为单一 unified cache K_u ∈ R^{N×b×d}，物理块总数增加 l×H 倍。

该重构的核心动机是支持 variable-head-rate eviction：不同 head 可以有不同的 allocated block 数量（因为每个 head 独立分配/释放 blocks），被 evicted 的 blocks 可以直接释放物理内存而非仅增加碎片。

从系统架构角度拆解术语：

**Block Layout 变迁（以 Llama-3.1-8B 为例，l=32, H=8, b=16）**：

```
# Vanilla vLLM PagedAttention
physical K cache: K = [K^(0), K^(1), ..., K^(31)]  # 32 per-layer tensors
each K^(m): shape [N, 8, 16, d]  # N blocks, 8 heads, 16 tokens, d-dim
block table T: shape [B, L_max/16]  # 共享跨所有 layers 和 heads
# To get KVs for layer m, head h, sequence s, token position i:
block_n = T[s, i/16]
offset = i % 16
k = K^(m)[block_n, h, offset, :]  # all heads share block_n

# KV-Compress PagedAttention
unified K cache: K_u shape [N*total, 16, d]  # single unified tensor
# total = l*H = 32*8 = 256x more blocks
block table T: shape [B, 32, 8, L_max/16]  # per-layer per-head
# To get KVs for layer m, head h, sequence s, token position i:
block_n = T[s, m, h, i/16]
offset = i % 16
k = K_u[block_n, offset, :]  # per-head block lookup
```

**Variable-Head-Rate 内存释放在两种布局下的差异**：
```
# Vanilla layout: evict head h from layer m
# → block_n 仍被其他 heads 使用 → 无法释放 block
# 仅增加该 block 内该 head 位置的"空洞"

# KV-Compress layout: evict head h from layer m
# → 该 head 的所有 blocks 独立分配
# → 释放该 head 的 E blocks → block manager 回收物理内存
```

术语一般如何实现？如何使用？

实现涉及：(1) vLLM 的 block manager 修改——从 shared block index 改为 per-head per-layer block table；(2) physical cache 从 l 个 tensor 改为 single unified tensor；(3) PagedAttention kernel 需能通过 per-head block table 索引（而非统一的 shared index）。KV-Compress 在 vLLM v0.6.0 上实现（开源 https://github.com/IsaacRe/vllm-kvcompress）。

适用场景：需要 variable-head-rate KV eviction 的任何 paged-attention 框架。代价是 block table 大小增长 l×H 倍，以及 block 数量增加 l×H 倍——需配合 GPU 端 block 管理器以避免 CPU 调度瓶颈。

涉及论文标题：
- KV-Compress__Paged_KV-Cache_Compression_with_Variable_Compression_Rates_per_Attention_Head

---

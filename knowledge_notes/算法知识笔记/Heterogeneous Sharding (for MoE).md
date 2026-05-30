## Heterogeneous Sharding (for MoE)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Heterogeneous Sharding 是 Hecate 在 FSSDP 的 sharding phase 中使用的跨层 MoE shard 分配算法 (Algorithm 2)。与传统的 homogeneous sharding（每个 device 等量分配 expert——如 64 experts / 32 devices = 每 device 2 experts）不同，heterogeneous sharding 允许每个 MoE shard 包含 0 到 |ℰ| 个任意数量的 expert，且不同 MoE layer 可以有不同的分配方案，只要所有 layer 的 shards 在每 device 上的总内存需求均衡。

设计动机：sparse materialization 主要帮助 overloaded expert（物化到多个 device 分散负载），但 underloaded expert 的 placement 也需要优化。例如，若某 node 上的所有 MoE shards 只包含 underloaded experts，则该 node 的入站带宽可能在 All-to-All 中被这些 crowded underloaded experts 的 token 淹没，因为该 node 可能是这些 token 的唯一目的地。Heterogeneous sharding 通过跨层统一调度 underloaded experts 来缓解这一问题。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Algorithm 2 的核心思路：
1. 将各层 experts 分为 J (overlappable/top-t overloaded) 和 J' (underloaded/其余) 两个不相交集
2. 先放置 J' (underloaded experts)：layer by layer，优先最 overloaded 的 layer（因为 overloaded expert 更多的 layer 面临的 All-to-All congestion 更严重），每个 expert 选 least-loaded node → least-loaded device on that node
3. 再填充 J (overloaded experts)：任意分配到剩余 slots

```
Algorithm 2: Heterogeneous Sharding
Input: F^g (所有层 expert load), t (overlap degree)
Output: P^g = {P_0, ..., P_L} (各层 sharding plan)

J ← top-t experts by load for each layer
J' ← E^g - J
slots_per_device ← |E^g| / |D|

// Phase 1: Place underloaded experts
L ← {E_l ∩ J' for l = 0..L}  // 各层的 underloaded expert set
for each E'_l in sortByMaxLoadDescending(L):
    P_l ← ∅
    for each e in sortByLoadDescending(E'_l):
        n ← least-loaded node (优先剩余 slots 少的)
        d ← least-loaded device on node n (同上优先级)
        P_l ← P_l ∪ {(d, e)}
        S_d ← S_d - 1
    P^g ← P^g ∪ P_l

// Phase 2: Fill remaining slots with overlappable experts
update P^g by arbitrarily placing J into remaining slots
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- Heterogeneous sharding 涉及跨 MoE layer 的状态迁移（re-sharding），会引入 critical path 上的延迟。但 Hecate 论证 re-sharding 可以低频触发（每 100 iterations），因为 underloaded experts 的梯度更新幅度小（处理的 token 少），其 load 变化缓慢（图 3 证实的 temporal locality）。实验显示 heterogeneous sharding 在不同 re-sharding 间隔（10-100 iterations）下均能提供一致的 1.34-1.42× speedup，证明对频率不敏感。
- Re-sharding 仅在 shard 确实发生变化时才执行实际的数据迁移，进一步摊销开销。
- 与 sparse materialization 的组合使用是关键：单独使用 heterogeneous sharding 或 sparse materialization 的效果有限，两者结合能实现 3.32× speedup。

涉及论文标题：
- Hecate: Unlocking Efficient Sparse Model Training via Fully Sharded Sparse Data Parallelism

## Batched Patch Cache Operations

术语是什么？通过联网搜索让回答具体和精准。

Batched Patch Cache Operations是MixFusion中用于高效管理patch级缓存的批量操作机制。由于patch-level caching需要在每个denoising step的每个block前后进行cache query（判断哪些patch可复用）/update（更新重算patch的结果）/insert（新增patch）/delete（清理退出GPU的patch），per-patch逐一操作会产生巨大开销——在SD3中每step约40-50ms含24 blocks，每个block的cache操作必须<2ms才能获得净收益。Batch coalescing将同一block中的所有patches的cache操作聚合为一次batch调用，通过三集合分类（Common Set/New Set/Expired Set）并行处理，amortize per-patch overhead。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Batched Patch Cache Operations的执行流程：

```
Cache系统以map结构存储（patch unique ID → cached_data），每个block维护独立cache。

输入：patch_indices[N] + intermediate_results[N]（N=当前batch的patch数）

1. Compare（索引比对）：
   input_set = set(patch_indices)
   cache_set = set(cache_pool.keys())
   common_set = input_set ∩ cache_set   // IDs同时存在→需verification
   new_set = input_set - cache_set      // IDs仅在input→插入
   expired_set = cache_set - input_set  // IDs仅在cache→删除

2. Slice（分片处理）：
   对common_set中每个patch：
     if MSE(intermediate_result, cached_data) > threshold:
         标记为recompute（mask=0）
     else:
         标记为reuse（mask=1）
   → 生成reuse_mask[N]

3. Compose（输出组合）：
   对common_set中reuse的patch：
     用cached_data替换intermediate_result对应位置
   对common_set中recompute的patch：
     保留新计算的intermediate_result
   → 生成masked_output[N]（所有position均有有效数据）

4. Update（批量更新）：
   对common_set中recompute的patch + new_set中所有patch：
     batch_insert(cache_pool, {id: new_data})
   对common_set中reuse的patch：
     batch_update_timestamp(cache_pool, ids)  // 仅更新时间戳
   对expired_set：
     batch_delete(cache_pool, ids)  // 对应patch已退出GPU
```

关键优化：(1) 所有cache操作在一次调用中完成→避免per-patch kernel launch overhead；(2) Common Set的verification（MSE comparison）使用vectorized GPU操作； (3) Expired Set检测使得cache自动清理退出GPU的patch，无需显式preemption支持。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MixFusion中用C++/CUDA实现batch cache manager：使用hash map存储patch ID到cached data的映射，batch操作通过GPU kernel并行处理（vectorized comparison, batch insert with coalesced writes）。Cache predictor（Random Forest, GPU端cuML）的输出reuse_mask直接作为batch cache operation的输入，形成predict→compute→combine→update的pipeline。batch size scaling study（Figure 17）显示cache management overhead随batch size modest增长（batch size 3→12, overhead仅增加约10%），验证了batching策略的scalability。

涉及论文标题：
- MixFusion: A Patch-Level Parallel Serving System for Mixed-Resolution Diffusion Models

---


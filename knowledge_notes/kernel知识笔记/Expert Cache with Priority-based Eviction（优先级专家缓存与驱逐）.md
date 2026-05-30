## Expert Cache with Priority-based Eviction（优先级专家缓存与驱逐）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Cache with Priority-based Eviction 是 FineMoE 在 GPU memory 中管理 expert weights 的缓存机制，包括两个优先级计算：**prefetching priority**（决定哪些 experts 优先从 CPU 加载到 GPU）和 **eviction priority**（决定 GPU cache 满时驱逐哪些 experts）。两种 priority 均基于 searched expert map 中的概率分布 p_{l,j} 计算，使 GPU cache 在有限的显存约束下（6GB-96GB）最大化 expert hit rate。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
Expert Prefetching Priority:
PRI^{prefetch}_{l,j} = p_{l,j} / (l - l_now)

其中:
  p_{l,j}: searched expert map 中 expert j 在 layer l 的被选概率
  l - l_now: 距离当前 layer 的层数
  → 近层 (small l-l_now) + 高概率 (large p) = 高优先级

Expert Eviction Priority:
PRI^{evict}_{l,j} = 1 / (p_{l,j} × freq_{l,j})

其中:
  p_{l,j}: searched expert map 中的概率
  freq_{l,j}: expert 被访问的历史频率 (LFU-like)
  → 低概率 (small p) + 低频 (small freq) = 高 eviction 优先级 (先被踢出)

GPU Cache 管理流程（每次专家访问时）:

# Prefetching
for each E_{l,j} in E_prefetch:
    priority = p_{l,j} / (l - l_now)
    插入 GPU task pool (按 priority 排序的 priority queue)

# GPU Task Pool 异步执行:
while task_pool not empty:
    task = pop_highest_priority()
    cudaMemcpyAsync(host_ptr→device_ptr, expert_size, stream=prefetch_stream)
    ExpertCache[expert_id] = device_ptr

# Eviction (Cache 满时)
while GPU_cache_memory > budget:
    worst_expert = argmax_{E_{l,j} in cache} 1/(p_{l,j} × freq_{l,j})
    cudaFree(ExpertCache[worst_expert])
    从 ExpertCache 中移除 worst_expert

# On-Demand Loading (Expert Miss, 最高优先级)
if expert_not_found in ExpertCache:
    暂停所有 pending prefetch tasks
    cudaMemcpyAsync(host→device, expert_weights, stream=on_demand_stream)
    synchronize(on_demand_stream)  # 等待加载完成
    恢复 prefetch tasks
```

关键设计选择：
- 不使用 LRU (Least Recently Used)：expert usage 是 layer-wise sequential 的（一层接一层），"recently used" 的 experts 不会再被近期使用（因已跳过该层），LRU 不适合 expert offloading 场景。
- LFU + probability：在 MoE-Infinity LFU 基础上集成 expert map probability，使高频 + 高概率 experts 获得最强缓存保护。
- On-demand loading 可抢占 prefetch：expert miss 直接影响 forward 能否执行，优先级高于 speculative prefetch。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FineMoE 基于 MoE-Infinity 代码库，Expert Cache 以 C++ CUDA Runtime API 实现。GPU task pool 使用 C++ 异步线程 + CUDA streams 管理。Expert to GPU mapping 使用 hash map (Python dict → C++ map)，multi-GPU 场景下按 round-robin 分配 experts 到不同 GPUs 以均衡负载。Prefetch/eviction priority 在每次 expert map search 完成时更新，确保缓存策略实时反映最新的 prediction confidence。

实验表明 FineMoE 的 priority-based caching 在 expert hit rate 上超越纯 LFU 和 LRU（图 14b 消融实验）。在 limited GPU cache（6GB）场景下效果最显著：TPOT 比 MoE-Infinity (LFU) 降低 29%。

涉及论文标题：
- Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading

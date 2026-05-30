## Dependency-aware Expert Management (依赖感知专家管理)

术语解释
Dependency-aware Expert Management 是 CoServe 提出的 CoE 专家淘汰策略，当 GPU 内存不足需为所需 expert 腾出空间时，利用专家依赖关系和使用概率做出更优的淘汰决策。

术语是什么？
该策略包含两阶段淘汰：
1. **Stage 1 — 依赖感知淘汰**：优先淘汰无前置依赖（preliminary dependency）的后续 expert。因为这些 expert 在前置 expert 加载完毕前无法执行，留在 GPU 中造成内存浪费。按显存占用降序逐一淘汰，直至满足新 expert 的内存需求。
2. **Stage 2 — 使用概率淘汰**：若 Stage 1 淘汰所有符合条件的 expert 后仍不够，按预评估的使用概率升序淘汰（概率最低的优先淘汰）。

与 MoE 中 LRU 淘汰的区别：LRU 仅依赖历史访问时间，在 CoE 中不够准确；CoE 可从路由规则直接计算使用概率，比历史统计更精确。

从系统架构角度拆解术语：
CoServe 中 Expert Management 运转流程：
```
需要加载 Expert_new (memory_footprint = M_new):
available = GPU_free_memory
if available >= M_new: load directly

// Stage 1: Dependency-aware eviction
for expert in model_pool (sorted by dependency_chain):
  if expert 无前置依赖 or 前置依赖 expert 不在 model_pool:
    candidates.append(expert)
sort candidates by memory_footprint desc
while available < M_new and candidates:
  evict candidates[0]
  available += candidates[0].memory
  candidates = candidates[1:]

// Stage 2: Probability-based eviction
if available < M_new:
  sort remaining experts by usage_probability asc
  while available < M_new:
    evict experts[0]
    available += experts[0].memory
```

术语一般如何实现？如何使用？
- 实现于 CoServe 的 Expert Manager 模块
- 使用概率来自 Offline Profiler 阶段（从路由规则或小样本数据集计算）
- 使用前提：CoE 系统路由规则可离线分析
- 管理开销 < 0.2% 总任务时间

涉及论文标题：
- CoServe: Efficient Collaboration-of-Experts (CoE) Model Inference with Limited Memory

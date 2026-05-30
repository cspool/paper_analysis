## Three-Phase Hierarchical Expert Prefetching（三阶段分层专家预取）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Three-Phase Hierarchical Expert Prefetching 是 MoE-SpeQ Expert Scheduler 的缓存管理策略，利用 ELB 多步 lookahead 将 VRAM 从反应式 LRU cache 转为 proactive staging area。三阶段递进利用 ELB 不同时间跨度的预测：Phase I 利用近期高确定性（ELB 前部）、Phase II 利用中期高置信度（ELB 中部）选择性预取、Phase III 饱和 VRAM（ELB 尾部全部缺失 experts）。结果：cache hit ratio 96.25-99.85%（vs LRU 29.2-76.56%）。

从系统架构角度拆解术语：
```
# Phase I: Cache Priming — 利用时序局部性，不发起 PCIe transfer
serve_from_cache(ELB[0:2])

# Phase II: Adaptive Prefetch — 高 confidence 条目选择性预取
async_prefetch(ELB[2:k-2] where confidence > threshold)

# Phase III: Cache Saturation — 全部缺失 experts 高优先级预取
high_priority_prefetch(ELB[k-2:k])
```
Phase 边界随 draft progress 动态调整，prefetch 使用独立 CUDA H2D stream。lookahead-aware eviction 优先淘汰 ELB 中无后续需求的 cached experts。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
依赖 CUDA multi-stream（prefetch H2D stream 与 compute stream 并行）+ CUDA events 同步。效果（DeepSeekV2-Lite, 16/24/32GB cache）：speculative 96-99% hit rate 远超 LRU、LRU(scaled) 和 Single Prefetch。论文未开源。

涉及论文标题：
- MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading for Mixture-of-Experts

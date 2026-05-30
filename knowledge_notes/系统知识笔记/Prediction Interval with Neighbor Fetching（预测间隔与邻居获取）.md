## Prediction Interval with Neighbor Fetching（预测间隔与邻居获取）

术语解释
一种摊销稀疏注意力预测器开销的系统优化策略，通过周期性（而非每步）运行 token 重要性预测器并在选择结果上扩展空间邻居来补偿重要性漂移。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Prediction Interval with Neighbor Fetching 是 TokenButler 提出的双层开销摊销机制。第一层 **Prediction Interval**：预测器每 N 步（而非每步）运行一次完整的重要性评分 pipeline（importance query prediction → score computation → top-B selection → KV gather），中间 N-1 步直接复用上一轮的 stale selection。这使预测器调用次数从 T 降至约 T/N。第二层 **Neighbor Fetching**：为补偿 stale selection 导致的重要性漂移，对每个选中的 token 同时获取其空间邻居 token（利用 consecutive tokens 通常携带相关信息的特点，如多 token 实体或推理链）。使用 cluster-aware algorithm：连续选中的索引形成 cluster，每个 token 的 neighbor 放置在 cluster 边界之外以最大化覆盖。总选中位置数 = 2B（B 为原始 budget）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 LLM 推理系统中的执行流程：

```
Decode Loop (系统视角):
    for step t in generation:
        # Step 1: KV Cache 管理
        新 token 进入 Local Window Buffer (循环队列)
        if local_window_full:
            批量投影最近 N 个 token 的 keys → K_proj (deferred projection)
        
        # Step 2: 重要性评分（仅触发步）
        if t % N == 0:                          # Prediction Interval
            Q_imp = predictor(hidden_states)     # MLP forward
            for each consumer layer:
                scores = Q_imp @ K_proj_cache.T  # 低维评分
                # Neighbor Fetching
                selected_raw = topk(scores, B)
                selected_expanded = cluster_aware_neighbor_expand(selected_raw)
                # → 2B unique positions
            migrate_to_important_buffer(selected_expanded)
        
        # Step 3: Attention (每步执行)
        K_attn = [Sink(128) | Important(2B) | Local_Window(256)]
        attn_out = FlashAttention(Q, K_attn, V_attn)
        
        # Step 4: 生成下一个 token
        next_token = sample(attn_out @ W_out)
```

系统关键设计决策：
- N=16 时预测器调用减少 16×，RULER 精度仅下降 1.1% (89.93% → 88.80%)
- Sparse budget B 固定（如 8K tokens），Neighbor Fetching 后 = 2B，Important Buffer 需按 2B 分配
- Local Window Buffer 每步更新（不受 interval 影响），保证最新 token 始终可用
- 延迟投影：token 在 window 中停留的 N 步内无需投影 key，充分利用 Local Window 的 dense attention 覆盖

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：
1. Cluster-Aware Neighbor Expansion：扫描 sorted selected indices，连续相邻的归为一个 cluster；每个 token 的 neighbor 放置在 cluster 右边界之后（避免 cluster 内部重复），neighbor 本身也可能是连续索引从而形成新的扩展。
2. Stale Selection 复用：中间步直接跳过 predictor forward 和 score computation，仅更新 Local Window（O(1) 操作）。Important Buffer 内容保持静态直至下次预测步。
3. Deferred Key Projection：利用 token 在 local window 中自然被 dense attention 覆盖的 N 步窗口期，批量执行 key 投影 GEMM K[N, D] @ W_K[D, d'] → K_proj[N, d']，显著提升 GPU 利用率（从每步 1 个 token 的 projection 变为每 N 步 N 个 token 的批量 projection）。
4. 效果：on-GPU 场景，N=16 时 1.6× speedup over Dense Attention @ 128K context；CPU offloading 场景（>=256K），bottleneck 在数据传输而非预测器计算，N=1 已接近 Oracle，N 增大收益有限。

涉及论文标题：
- TokenButler: Token Importance is Predictable

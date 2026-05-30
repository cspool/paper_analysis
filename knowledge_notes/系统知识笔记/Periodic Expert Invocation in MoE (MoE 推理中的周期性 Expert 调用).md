## Periodic Expert Invocation in MoE (MoE 推理中的周期性 Expert 调用)

术语解释
Periodic Expert Invocation 是一种策略，在此策略下 expert 预测器不是每个推理请求都调用，而是每隔固定数量的 prompts（如 p=40）才调用一次。在两次调用之间，所有推理请求复用当前已加载在 GPU 上的 expert 集合。该策略在维持模型输出质量（perplexity）的同时大幅 amortize expert 预测和加载的开销。

术语是什么？
eMoE 发现 consecutive prompts 间存在高相关性（cross-correlation 0.48-0.55），且在 ≤60 prompts 内复用同一组 expert 对 perplexity 影响极小。基于此观察，Periodic Expert Invocation 设定一个 periodic interval p（实验确定为 40），仅在每第 p 个 prompt 时：

1. 调用 Expert Predictor（BERT-XLNet）预测新的 expert 集合
2. 比较预测结果与当前 GPU 上的 expert，加载新 expert、卸载未命中 expert
3. 对中间 p-1 个 prompts，直接复用已加载 expert 进行推理

从系统架构角度拆解术语：

```
=== Periodic Expert Invocation Flow ===

System State:
  request_index = 0, 1, 2, ... (global counter)
  p = 40 (period, determined empirically from perplexity experiment)
  gpu_experts = {} (currently loaded experts per layer)

For each incoming request (request_index i):
  
  if i == 0 (first request):
    Load ALL experts to GPU (cold start)
    Process request → generate output
    request_index++
  
  elif i % p == 0:
    ===== Predictor Invocation =====
    1. Call Expert Predictor:
       future_experts = predict(expert_history)
    2. For each MoE layer:
       new_experts = future_experts[layer] ∖ gpu_experts[layer]
       evict_experts = gpu_experts[layer] ∖ future_experts[layer]
       async_load(new_experts)      # CPU→GPU PCIe
       async_evict(evict_experts)   # GPU→CPU
    3. Update gpu_experts = future_experts
    ===============================
    Process request with updated gpu_experts
    request_index++
  
  else:
    # Reuse: no predictor call, no expert loading
    Process request with current gpu_experts
    request_index++

Time Overhead (amortized over p prompts):
  eMoE-A: predictor call ~0.334s (Mixtral) / ~0.381s (OpenMoE)
          → 0.24%-0.47% of average inference time per request
  eMoE-L: predictor call ~4.211s (Mixtral) / ~1.387s (OpenMoE)
          → 1.69%-3.11% of average inference time per request
```

术语一般如何实现？如何使用？
- 周期 p 的选择需要 empirical tuning：balanced between accuracy degradation（p 过大 → perplexity 上升）和 latency overhead（p 过小 → 频繁预测/加载）
- eMoE 实验中 p=40 基于 perplexity-vs-reuse 实验（Fig 10）：perplexity 在 20-40 prompts 保持平稳，60+ prompts 开始上升
- 适合具有 temporal coherence 的 workload（如对话、系列相关问题），对完全随机/独立请求效果可能变差
- 可与 task-aware expert loading 结合：在 periodic invocation 时，对 routing-insensitive 任务仍然跳过 expert loading

涉及论文标题：
- eMoE: Task-aware Memory Efficient Mixture-of-Experts-Based (MoE) Model Inference

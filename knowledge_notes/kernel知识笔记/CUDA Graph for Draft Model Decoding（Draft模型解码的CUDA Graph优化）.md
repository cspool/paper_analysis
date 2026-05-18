## CUDA Graph for Draft Model Decoding（Draft模型解码的CUDA Graph优化）

术语是什么？通过联网搜索让回答具体和精准。
CUDA Graph 是 NVIDIA CUDA 提供的机制，允许将一系列 kernel launch 和 memory copy 操作预录制为有向无环图（DAG），后续通过单次 graph launch 回放整个图，消除逐个 kernel launch 的 CPU-GPU 同步和 driver overhead。AdaServe 将 CUDA Graph 用于 draft model 的 speculative decoding steps：从第二个 speculation step 到第 d 步，若活跃请求数相同，则每步计算形状一致，可复用预捕获的 CUDA graph。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
AdaServe 中 CUDA Graph 的使用流程：
```
// 预捕获阶段（系统启动时或活跃请求数变化时）
for num_requests in possible_range:
    graph[num_requests] = cudaGraphCreate()
    cudaStreamBeginCapture(stream)
    // 录制 d-1 步 draft decoding（step 2 到 step d）
    for step = 2 to d:
        draft_model.forward(tokens, num_requests)  // 固定 batch size
    cudaStreamEndCapture(stream, graph[num_requests])
    executable[num_requests] = cudaGraphInstantiate(graph[num_requests])

// 运行时（每轮 iteration）
// Step 1: 首次 draft forward（形状可能变化，单独 launch）
draft_model.forward_first(tokens, current_num_requests)

// Steps 2-d: 复用 CUDA Graph
if current_num_requests == prev_num_requests:
    cudaGraphLaunch(executable[current_num_requests], stream)  // 单次 launch
else:
    // 活跃请求数变化 → 重新捕获或用普通 launch
    fallback_individual_launch()
```
关键优化点：(1) 从第二步开始的 draft decoding 有 d-1 步形状相同→可录制成单图；(2) 单次 graph launch 替代 d-1 次独立 kernel launch → 显著减少 CPU-GPU sync overhead；(3) 活跃请求数变化时才重新捕获 graph。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 AdaServe 中实现于 FlexFlow Serve 的 execution engine 中。CUDA Graph 优化使 draft model 的重复 decoding steps 的 kernel launch overhead 大幅降低。实验显示 CPU selection overhead 仅占总 serving time 的 0.41%/0.31%，说明 draft decoding 的 GPU 执行 overhead 已被有效控制。

涉及论文标题：
- AdaServe: Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding


---


## Asynchronous Expert Prefetching in GPU Task Pool（GPU 异步专家预取任务池）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Asynchronous Expert Prefetching in GPU Task Pool 是 FineMoE 的 GPU-side 专家预取执行机制：Expert Map Searcher 确定需要预取的 expert 集合后，将 prefetch 任务提交到 GPU space 的 task pool（priority queue），由异步线程调度 CUDA async memory copy 将 expert weights 从 CPU 传输到 GPU。任务按 prefetching priority 排序执行。关键设计：prefetch tasks 与 inference computation 使用独立的 CUDA streams，使 CPU→GPU 数据传输与 forward pass 重叠执行。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
GPU Task Pool 执行流程（CUDA Stream 级别）:

CUDA Stream 分配:
  stream_compute: inference forward pass (attention, gate, expert GEMM)
  stream_prefetch: async CPU→GPU expert weight transfer
  stream_ondemand: emergency on-demand loading (expert miss)

Task Pool 数据结构:
  task_pool = PriorityQueue[
    {expert_id: "l_j", priority: p/(l-l_now), size: expert_weight_bytes, action: "prefetch"},
    ...
  ]

异步执行流程:
┌─ stream_compute ──────────────────────────────────────────────┐
│ Layer 1: attn → gate → expert compute                        │
│ Layer 2: attn → gate → expert compute                        │
│ Layer 3: attn → gate → expert compute                        │
│ ...                                                           │
└───────────────────────────────────────────────────────────────┘
    ▲ (不等待 prefetch 完成)

┌─ stream_prefetch ─────────────────────────────────────────────┐
│ cudaMemcpyAsync(host→dev, expert_L4_w0, stream_prefetch)     │
│ cudaMemcpyAsync(host→dev, expert_L5_w1, stream_prefetch)     │
│ ...                                                           │
└───────────────────────────────────────────────────────────────┘

Expert Miss 处理 (抢占机制):
  if 当前 layer 需要的 expert 不在 cache:
    1. 暂停 stream_prefetch 上的所有 pending tasks
    2. 在 stream_ondemand 上: cudaMemcpyAsync(host→dev, missed_expert)
    3. synchronize(stream_ondemand) → forward 该层
    4. 恢复 stream_prefetch 上的 pending tasks
```

与 MoE-Infinity synchronous prefetching 对比：
- MoE-Infinity: 每层 forward 前同步等待 expert prediction + prefetch → prefetch latency 直接加在 critical path
- FineMoE: forward 与 prefetch 异步重叠 → prefetch latency 不进入 critical path → overhead <1% iteration time

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FineMoE 的 GPU task pool 基于 MoE-Infinity C++ CUDA Runtime API 实现。Multiple CUDA streams 用于分离 compute 和 data transfer。Task priority queue 使用 C++ std::priority_queue 实现，按 PRI^{prefetch} = p/(l-l_now) 排序。On-demand loading 通过 flag 机制抢占：设置 global flag 暂停 prefetch dispatcher，等待 on-demand load 完成后清除 flag 恢复。此设计使 FineMoE 即使在高 expert miss rate 场景下也能最小化 on-demand loading 对 critical path 的影响。

涉及论文标题：
- Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading

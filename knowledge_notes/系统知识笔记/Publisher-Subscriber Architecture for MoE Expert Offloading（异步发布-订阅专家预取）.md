## Publisher-Subscriber Architecture for MoE Expert Offloading（异步发布-订阅专家预取）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Publisher-Subscriber Architecture 是 FineMoE 用于解耦 expert map searching / expert prefetching 与 inference forward pass 的异步通信架构。Inference process 作为 Publisher 持续将 context 数据（semantic embeddings + expert probability distributions）写入 Expert Map Store（作为 Message Broker），Expert Map Searcher 作为 Subscriber 订阅这些 context 数据并异步执行 similarity search 和 expert prefetching。这种设计避免了 MoE-Infinity synchronous prefetching 中"每层 forward 前必须等待 expert prediction + prefetch 完成"的 blocking 瓶颈。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
Publisher-Subscriber 消息流（FineMoE 一次 iteration）：

[Inference Process — Publisher]
   │ 每层 forward: attention → gate → expert computation
   │ 同时写入 context 到 Expert Map Store (Broker):
   │   - semantic_embedding[tokens] → tensor write
   │   - expert_trajectory[layer] → tensor write
   ▼
[Expert Map Store — Message Broker]
   │ CPU memory 中的 ndarray 存储
   │ 持续接收新 context，发送通知到 Subscriber
   ▼
[Expert Map Searcher — Subscriber]
   │ 异步接收 context → cosine similarity search
   │ 返回最高分 historical expert map
   │ 发出 prefetch 指令到 GPU task pool
   ▼
[GPU Task Pool]
   │ CUDA stream async copy queue
   │ cudaMemcpyAsync(host→device, expert_weights)
   │ 不阻塞 inference process 的 CUDA stream
   ▼
[Expert Cache — GPU Memory]
   │ 接收 prefetched experts → hash map 更新
   │ Forward pass 查 cache → hit: 直接用; miss: on-demand load
```

关键优势：
- Map searching + prefetching overhead 不进入 inference critical path
- GPU task pool 异步线程调度 prefetch/on-demand loading
- Expert miss 时暂停所有 pending prefetch → 立即 on-demand load → 恢复 prefetch
- Overhead < 1% iteration latency (< 50ms for all non-async ops)

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FineMoE 中 Expert Map Store 用 PyTorch/NumPy ndarray 实现 broker 功能。异步执行通过 CUDA streams 分离：main inference stream 执行 forward pass，prefetch stream 执行 CPU→GPU memory copy。GPU task pool 使用 C++ threads 调度，on-demand loading 拥有最高优先级（可抢占 pending prefetch tasks）。此设计使 FineMoE 即使 prefetch distance 较小（d=3 for Mixtral）也能有效隐藏 prefetch latency。

涉及论文标题：
- Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading

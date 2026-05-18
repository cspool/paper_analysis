## Iteration-Level Scheduling（迭代级调度）

术语是什么？通过联网搜索让回答具体和精准。
Iteration-Level Scheduling是当前LLM serving的主流调度范式，以一次完整模型forward pass（穿过所有transformer layers）作为最小调度和批处理单元。Orca提出continuous batching后，iteration-level成为de facto标准：系统在每个iteration开始时决定哪些请求参与本轮batch，整轮执行全部N层后，在下一iteration加入新请求或移除已完成请求。Sarathi-Serve引入chunked prefill（长prefill拆分为chunks交错在decode iteration间），DistServe引入prefill-decode disaggregation（prefill和decode分配到专用instance），但各instance内部仍使用iteration-level scheduling。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Iteration-level scheduling的执行模型：
```
// 每轮iteration的标准流程
for iteration t = 1, 2, ...:
    // 1. 调度决策（仅iteration开始时）
    batch = select_requests(active_requests, waiting_requests, scheduling_policy)
    // 2. 执行完整forward pass（N layer, 不可中断）
    for layer j = 1..N:
        for r in batch:
            hidden[r] = transformer_layer[j](hidden[r], kv_cache[r])
    // 3. 生成token
    for r in batch:
        if is_decode(r):
            output_token[r] = sample(project_to_vocab(hidden[r]))
    // 4. 仅此时可加入新请求或移除已完成请求
```

在multi-SLO场景下的限制：(1) Prefill：chunk size trade-off不可逃避——大chunk（3200 tokens）per-token latency比小chunk（100 tokens）低45.4%但per-iteration latency高16×；一旦chunk开始执行必须完成所有层才能切换，导致head-of-line blocking。(2) Decode：所有请求以统一batch size和相同节奏执行，被迫服从最严格SLO（如50ms TBT），无法利用relaxed请求的slack（可容忍7×更多并发）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
vLLM、SGLang等主流serving框架的默认调度模式。vLLM的scheduler每轮从waiting queue取请求加入running batch，等待所有请求完成当前step后再进行下一轮调度。Chunked prefill在此基础上拆分长prefill为固定大小chunk，但chunk粒度仍是iteration级别。Laser识别iteration-level是multi-SLO场景的性能瓶颈，提出layer-level scheduling作为细粒度替代方案。

涉及论文标题：
- Laser: Unlocking Layer-Level Scheduling for Efficient Multi-SLO LLM Serving
- Sarathi-Serve: Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve
- DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving

---

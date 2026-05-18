## Layer-Level Scheduling（层级调度）

术语是什么？通过联网搜索让回答具体和精准。
Layer-Level Scheduling是Laser提出的LLM serving调度机制，将调度粒度从完整模型forward pass（iteration）细化到单个transformer layer。传统的iteration-level scheduling在每轮前向传播穿过所有N层后才进行调度决策（切换、合并、批处理请求），而layer-level scheduling在每层边界都可以做出调度决策，提供N倍的调度机会和更精细的执行控制。Laser的两阶段实现：prefill侧layer-level chunked prefill（可在layer边界抢占/合并prefill chunk），decode侧layer-level decode batching（可通过L/O参数控制每个请求每iteration执行的层数）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Layer-level scheduling的执行流程（以prefill抢占为例）：
```
给定: Model with N transformer layers
      执行中的chunk C (requests R_c) 正执行到layer k
      新请求r_new到达

Scheduler:
  slack(r_new) = TTFT_target(r_new) - estimated_remaining_prefill_time(r_new)
  remaining_iter_time = Σ_{j=k}^{N} estimated_layer_latency(j, C)
  
  if remaining_iter_time > slack(r_new):
      // 当前chunk剩余时间会危及新请求SLO → 抢占
      Executor.save_intermediate_state(C, layer=k)  // 存入GPU intermediate cache
      if can_merge(r_new, C, from_layer=k):
          forward r_new to layer k
          merge r_new with C from layer k+1  // 合并执行提升utilization
      else:
          prioritize r_new alone  // 单独优先执行
          enqueue C for later restoration
  else if queue_is_empty():
      // 无需抢占且无需排队 → 尝试动态合并新请求到当前chunk
      partition r_new into chunk_sized_to_meet_its_TTFT
      forward r_new to current layer
      merge r_new_chunk with C from current layer  // 动态增大有效chunk

  // 队列调度（EDF原则）
  sorted_queue = EDF_sort(pending_requests)  // 按 arrival_time + TTFT_target 排序
```

与iteration-level的关键区别：iteration-level必须在完整N层后切换（锁定时间=完整iteration延迟，通常数十ms），layer-level可在任意layer边界切换（锁定时间=单层延迟，通常<1ms），大幅降低调度延迟和提高multi-SLO响应性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Laser在vLLM + Ray上实现layer-level scheduling。关键实现要素：(1) intermediate cache：GPU memory内存储layer-level hidden states，大小prefill 16384 tokens/decode 2048 tokens，Llama-70B上<256 MB；(2) fused CUDA kernel合并state caching和retrieval为单次操作；(3) offline latency profiling构建modular latency model（stateless模块用分段线性，stateful attention用线性，Pearson >0.78），profiling在<2秒完成；(4) 调度开销：request switching <1.5%，scheduling <3.8%（通过event-triggered updates和execution overlap与model计算重叠）。layer-level scheduling适用于共享foundation model、多应用多SLO的云serving环境。

涉及论文标题：
- Laser: Unlocking Layer-Level Scheduling for Efficient Multi-SLO LLM Serving

---

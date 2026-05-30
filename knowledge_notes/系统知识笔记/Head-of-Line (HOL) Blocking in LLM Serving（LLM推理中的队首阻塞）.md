## Head-of-Line (HOL) Blocking in LLM Serving（LLM推理中的队首阻塞）

术语是什么？
Head-of-Line (HOL) Blocking 原是网络交换中的经典排队现象——当队首数据包因目的地拥塞无法转发时，其后所有数据包被阻塞无法前进，即使它们的目的地空闲。在 LLM Serving 中，HOL blocking 指迭代级 FCFS 调度下，先到达的长序列 BE（Best-Effort）请求占据 batch 前排位置，导致后到达的 LS（Latency-Sensitive）请求被阻塞在整个完整 decode iteration（300-400ms for Mixtral 8×7B）之后才能被调度。

从系统架构角度拆解术语：
LLM serving 中的 HOL blocking 形成过程：

```
时间轴 → (Mixtral 8×7B, batch_size=32, A100 80GB)

t=0:     BE1-BE4 四个 decode 请求到达 → Scheduler 组 batch=[BE1,BE2,BE3,BE4]
         → Engine 开始执行 iteration (32 layers)

t=50ms:  LS1 到达 → Scheduler 收到，但 iteration 正在进行中
         → LS1 入队等待 iteration boundary

t=100ms: LS2 到达 → 同样等待

...

t=350ms: iteration 完成 → Scheduler 取回控制权
         → 停止 decode, prefill LS1, LS2
         → LS1 TTFT 含 300ms 排队延迟 (350-50ms)

总计: LS1 从 50ms 到达至 ~400ms 产生第一个 token
     → TTFT ≈ 350ms queuing + 50ms prefill = ~400ms
```

在 MoE 场景中 HOL blocking 因以下因素加剧：
1. **Decode iteration 时间长**：Mixtral 8×7B 的 32 层 × (attention + router + 8 experts) 导致单 iteration 300-400ms（比同规模 dense 更长）。
2. **Expert 负载不均**：hot expert（频繁被选中）执行时间长于 cold expert，barrier 同步使 iteration 时间被最慢 expert 决定。
3. **Batched expert execution**：batch size=32 时，每个 expert 可能需处理多个 token，线性增加 expert FFN 执行时间。

QLLM 的解决方式：通过 expert-level preemption 在任意 layer 的 router 后中断 BE batch，将 HOL blocking 的粒度从 iteration 级（300-400ms）降低到 layer 级（~10ms）。

术语一般如何实现？如何处理？
- **传统解决方案**（网络领域）：Virtual Output Queues (VOQ) —— 为每个输出端口维护独立队列，避免相互阻塞。QLLM 的 per-expert queue 设计与此理念一致（为每个 expert 维护独立队列）。
- **LLM Serving 解决方案**：
  - FastServe: token-level preemption via skip-join MLFQ，但需 recomputation。
  - Andes: preemptive scheduling with token-level granularity。
  - Llumnix: KV cache migration 实现负载均衡而不是抢占。
  - QLLM: expert-level preemption without recomputation（zero recomputation cost）。
- 核心权衡：preemption granularity（越细越好） vs preemption overhead（state save/restore cost，越细越大）。QLLM 的 Facade Pattern 实现 zero-copy state save/restore，压低 overhead 以支撑 expert-level 细粒度。

涉及论文标题：
- Priority-Aware Preemptive Scheduling for Mixed-Priority Workloads in MoE Inference

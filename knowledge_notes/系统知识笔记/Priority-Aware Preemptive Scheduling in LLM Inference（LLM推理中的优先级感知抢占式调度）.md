## Priority-Aware Preemptive Scheduling in LLM Inference（LLM推理中的优先级感知抢占式调度）

术语是什么？
Priority-Aware Preemptive Scheduling 是一种能够根据请求优先级（Latency-Sensitive vs Best-Effort）在 LLM 推理过程中随时中断低优先级任务、插入高优先级任务的细粒度调度机制。QLLM 将该思想应用于 MoE 模型，通过 expert-level 抢占突破传统 iteration-level scheduling 的 loop closure 限制——传统系统每 300-400ms 一个完整 iteration 才取回调度控制权，QLLM 在每个 MoE layer 的 attention/router 后都能通过 closed-loop feedback 重新调度，将 LS 请求的等待延迟从 300-400ms 降至单层执行时间（~10ms）。

从系统架构角度拆解术语：
QLLM 的优先级感知抢占调度流程（以 MoE layer 内部执行为例）：

```
// QLLM Scheduler 主循环
Scheduler:
  接收请求 → Dispatcher 按 priority 分派到四个队列:
    LS_PrefillQueue, LS_DecodeQueue, BE_PrefillQueue, BE_DecodeQueue

// Batch Engine 批次选择 (Algorithm 1)
function GetNextBatch(BatchSize):
    if LS_DecodeQueue.size() >= BatchSize:
        return LS_DecodeQueue.pop(BatchSize)       // 优先 LS decode
    elif !LS_PrefillQueue.isEmpty():
        batch = LS_PrefillQueue.pop(BatchSize)
        Fill from BE_PrefillQueue if space          // 填充剩余空间
        return batch
    elif !LS_DecodeQueue.isEmpty():
        batch = LS_DecodeQueue.pop()
        Fill from BE_DecodeQueue
        return batch
    elif !BE_DecodeQueue.isEmpty():
        return BE_DecodeQueue.pop(BatchSize)
    elif !BE_PrefillQueue.isEmpty():
        return BE_PrefillQueue.pop(BatchSize)

// Inference Engine 内部执行（Closed-Loop Feedback）
for layer L in 1..N:
    attention(batch, kv_cache)
    feedback_scheduler()                            // Checkpoint 1: attention后
    if LS_request_arrived:
        save_state(BE_batch)                        // Sequence对象保存BE状态
        execute_prefill(LS_request)                 // 立即执行LS prefill
        execute_decode_iteration(LS_request)        // LS decode一轮
        merge_into_batch(LS_request, BE_batch)      // Facade Pattern无缝合并
    router(batch)                                   // Top-K expert selection
    for each expert E:
        expert_FIFO_queue[E].process_tokens()       // Per-expert独立执行
    feedback_scheduler()                            // Checkpoint 2: router后
```

抢占的时序特征：在 A100 + Mixtral 8×7B 上，一个完整 decode iteration 需 300-400ms，而单层 attention+router 仅约 10ms。QLLM 将 LS 到达后的等待从 300-400ms 降低到约 10ms（当前层剩余执行时间），实现 ~30-40× 的调度延迟降低。

术语一般如何实现？如何使用？
- **QLLM 实现**：基于 HuggingFace TGI 构建，修改 MoE layer 插入 per-expert FIFO queues 和 closed-loop feedback hooks。用 Facade Pattern 的 Sequence/Batch 抽象替代传统 concat tensor，使得 per-sequence 状态可单独保存/恢复而无需 split-merge。Scheduler 通过四个优先级队列和 Algorithm 1 的 batch 选择逻辑实现优先调度。
- **TokenFlow 实现**（另一篇相关论文）：在 SGLang 上通过 buffer-aware scheduling 和 hierarchical KV cache management 实现 preemptive scheduling，但侧重 text streaming 场景。
- **通用框架要求**：(1) 支持请求优先级标记（LS/BE）；(2) 支持 inner-layer state save/restore（需要 Per-Sequence 抽象）；(3) 调度器可异步中断执行流（closed-loop feedback 或 interrupt-based）。
- 适用场景：数据中心混合工作负载（付费用户 LS vs 免费用户 BE）、交互式应用（chatbot LS）与批处理作业（文档摘要 BE）共存、需要 SLO 保证的推理服务。

涉及论文标题：
- Priority-Aware Preemptive Scheduling for Mixed-Priority Workloads in MoE Inference

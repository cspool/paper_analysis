## Per-Expert FIFO Queuing（每专家FIFO队列）

术语是什么？
Per-Expert FIFO Queuing 是 QLLM 提出的一种 MoE layer 内部 token 流转机制——每个 expert 维护一个独立的 FIFO 队列，router 将 token 的 sequence reference push 进对应 expert 的队列，各 expert 独立从其队列中 pop token 执行 FFN 计算。这打破了传统 MoE layer 的 barrier 同步模型（所有 token 必须等待所有 expert 完成后才进入下一层），使单个 token 可以在其两个 expert 都完成后立即推进，而不必等待 batch 中最慢的 expert。

从系统架构角度拆解术语：
传统 MoE layer（barrier 同步）vs Per-Expert FIFO Queuing：

```
# 传统 MoE layer (barrier 同步)
hidden = attention(x)
router_logits = gate(hidden)
for each token t:
    topk_experts[t] = topk(softmax(router_logits[t]), k=2)
# Barrier: 所有 token 被分配到所有 expert 的 batch 中
for each expert E:
    expert_input = concat([hidden[t] for t where E in topk_experts[t]])
    expert_output[E] = FFN[E](expert_input)
# Barrier: 等待所有 expert 完成
hidden_out = scatter_and_combine(expert_output, routing_weights)
# → 单层延迟 = max(expert_exec_time) 由最慢 expert 决定

# QLLM Per-Expert FIFO Queuing
hidden = attention(x)
router_logits = gate(hidden)
for each token t:
    topk_experts[t] = topk(softmax(router_logits[t]), k=2)
    for expert_id in topk_experts[t]:
        expert_queue[expert_id].push(sequence_ref[t])  // 轻量引用入队

# 各 expert 独立执行，无全局 barrier
for each expert E in parallel (conceptually):
    while expert_queue[E].not_empty():
        token_ref = expert_queue[E].pop()
        output = FFN[E](hidden_states[token_ref])
        sequence[token_ref].expert_outputs[E] = output
        sequence[token_ref].experts_completed += 1

# Token 级别的推进条件（非 batch 级别）
for each token t:
    if sequence[t].experts_completed == k:   // 该 token 的所有 k 个 expert 完成
        hidden_out[t] = combine(sequence[t].expert_outputs, routing_weights[t])
        advance_to_next_layer(t)              // 该 token 可立即进入下一层
```

Per-Expert Queue 的关键设计决策：
1. **存储 Sequence Reference 而非 tensor**：入队的是指向 Sequence 对象的指针/引用，而非完整 hidden state tensor。这避免了 tensor 拷贝开销，同时在 preemption 时 Sequence 状态自然一致。
2. **FIFO 策略**：先入队的 token 先被处理。结合 Scheduler 的优先级入队顺序（LS 优先 push），自然实现 LS token 在 expert 层面的优先执行。
3. **Partially vs Fully Processed Token 区分**：QLLM 维护 `experts_completed` 计数器，只允许所有 k 个 experts 都完成的 token 进入下一层。若 preemption 发生在 token 仅完成 1/2 experts 时，该 token 被标记为 partially processed，恢复时从其 Sequence 状态中读取剩余的 expert assignment。
4. **Queue Stall 防护**：当某 expert 的队列为空而其他 expert 队列积压时，该 expert 处于 idle 状态。QLLM 通过 Scheduler 层面的 batch composition 控制（优先 LS）减少 queue 不均衡。

术语一般如何实现？如何使用？
- **QLLM 实现**：在 HF TGI 的 MoE block 中替换原有 `torch.cat([hidden[t] for t in expert_tokens])` 为 per-expert `collections.deque`（Python FIFO queue）。由于 Python 层面而非 CUDA kernel 层面实现，适合原型验证但延展到生产需优化为 CUDA stream-level queuing。
- **性能含义**：在当前 A100 单 GPU 实现中，expert FFN 仍是顺序执行的（受 PyTorch CUDA stream 串行化限制），因此 per-expert queue 的主要收益不是并行化，而是解耦 batch-level barrier——允许 Scheduler 在 expert 粒度插入 preemption/context-switch。
- 适用场景：MoE 推理的细粒度调度、与 expert-level preemption 配合、未来 CUDA graph/MPS 支持 per-expert 并发时的并行加速。
- 推广到 Dense 模型：Dense 模型的 FFN 不区分 expert，per-expert queue 退化为 per-layer single queue，preemption 粒度变为 layer 级而非 expert 级。

涉及论文标题：
- Priority-Aware Preemptive Scheduling for Mixed-Priority Workloads in MoE Inference

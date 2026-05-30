## Expert-Level Preemption（专家级抢占）

术语是什么？
Expert-Level Preemption 是 QLLM 提出的一种以 MoE layer 内单个 expert 为粒度的任务抢占机制。不同于传统 iteration-level preemption（需等待完整 N 层 forward pass 完成），也不同于 token-level preemption（FastServe 的 skip-join MLFQ），Expert-Level Preemption 允许在任意 MoE layer 的 router 输出后中断 BE batch 中尚未被 expert 处理的 token，优先执行 LS 请求，然后从被中断点恢复 BE token 的执行——无需重新计算已完成的 attention 和 router 阶段。

从系统架构角度拆解术语：
Expert-level preemption 的执行流程与状态管理：

```
// Expert-Level Preemption 在单层内的执行
Layer L:
  // Phase 1: Attention (all tokens)
  hidden_states = self_attention(qkv_projections, kv_cache)
  // [preemption check point #1 — attention 不可中断但状态已保存]

  // Phase 2: Router
  router_logits = gating_network(hidden_states)
  topk_experts, routing_weights = topk(softmax(router_logits), k=2)
  for each token t:
      for expert_id in topk_experts[t]:
          enqueue(sequence_ref[t], expert_queue[expert_id])
  // [preemption check point #2 — router 完成，expert 尚未执行]

  // Phase 3: Per-Expert Execution (preemptable)
  for each expert E:
      while expert_queue[E].not_empty():
          token = expert_queue[E].pop()
          expert_output[token] = Expert_FFN[E](hidden_states[token])

  // Phase 4: Combine
  for each token t:
      hidden_states[t] = Σ routing_weights[t][j] · expert_output[t][j]
```

抢占发生时需保存的 per-token 状态（Sequence 对象维护）：
- `hidden_states`: 当前 layer 的 hidden states（已通过 attention，未通过 expert FFN）
- `routing_weights`: router 输出的 top-k expert 权重
- `expert_assignments`: router 选择的 expert IDs
- `per_expert_progress`: 哪些 experts 已完成执行、哪些尚未执行（distinguish fully/partially processed tokens）
- `kv_cache_entries`: attention 阶段写入的 KV cache

恢复时：BE token 从 Sequence 对象重新加载状态，Unified Dynamic KV Cache 恢复 cache 行，从被中断的 expert 处继续执行。

术语一般如何实现？如何使用？
- **QLLM 原型**：基于 HF TGI 构建。per-expert queue 用 Python list/FIFO 实现，Sequence 对象用 dataclass 封装所有 per-token 状态，Batch Facade 用 Python `__getattr__` 拦截 tensor 访问。Preemption 触发：Scheduler 检测到 LS 队列非空 → 通过 callback 通知 Engine → Engine 在下一个 preemption checkpoint（当前 layer 的 router 后）暂停 BE→保存状态→执行 LS→恢复 BE。
- **关键前提**：(1) Per-Sequence 独立状态存储（不能依赖 concat tensor）；(2) KV cache 支持 per-token 增量更新而非整体 rewrite；(3) Expert execution 是 per-expert 独立且幂等的（相同 input + expert weight 产生相同 output）。
- **与 Token-Level Preemption (FastServe) 的区别**：FastServe 通过 skip-join MLFQ 在 decode iteration 级别抢占——即如果某 token 的 decode 迭代数已超过其优先级分配的时间片，则加入下一优先级队列，需 recomputation。QLLM 的 Expert-Level Preemption 不丢弃任何计算——仅暂停，保存完整状态，恢复时 zero recomputation。
- 适用场景：MoE 模型 serving（因 expert 是天然的分界线，router→expert 间是自然的 preemption point），扩展到 dense 模型时以 layer 为 preemption point（router 不存在时）。

涉及论文标题：
- Priority-Aware Preemptive Scheduling for Mixed-Priority Workloads in MoE Inference

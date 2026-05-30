## Context-Coherent Expert Parallelism (上下文一致的专家并行)

术语解释
Context-Coherent Expert Parallelism 是 ExFlow 论文提出的一种 Expert Parallelism 变体：通过 AllGather 操作使所有 GPU 持有全部 token 的 context（包括初始 prompt 和每轮生成的新 token），消除传统 expert parallelism 中"token 必须回到原 GPU 执行 attention"的约束，从而将每个 MoE 层的 2 次 Alltoall（dispatch + combine）减少为 1 次 Alltoall（仅 dispatch），在 iteration 结束时补 1 次轻量 AllGather。

术语是什么？
传统 Expert Parallelism（如 DeepSpeed-MoE）结合了 Data Parallelism 和 Model Parallelism：DP 保证每个 GPU 的 token 和 context 是隔离的（互不可见），MP 保证每个 GPU 独占特定 expert。这导致 token 被 dispatch 到 expert GPU 后，必须在 MoE 层结束时通过 Alltoall combine 返回原 GPU，因为下一层的 attention 需要 token 与其 context 交互（context 仅在原 GPU 上）。

Context-Coherent Expert Parallelism 的核心洞察：GPT 推理是 auto-regressive 的——已生成的 token 是 immutable 的，仅作为 context 使用。因此，如果在推理开始和每轮迭代结束时将 context 同步到所有 GPU，token 就可以在任何 GPU 上原地执行 attention，无需返回原 GPU。

从系统架构角度拆解术语：
Context-Coherent Expert Parallelism 的执行流程：

```
=== 推理开始 ===
AllGather(all_contexts):
  GPU 0 广播自己的 g_0 个 context → 所有 GPU
  GPU 1 广播自己的 g_1 个 context → 所有 GPU
  ...
  GPU N-1 广播自己的 g_{N-1} 个 context → 所有 GPU
  # 结果: 每个 GPU 持有 sum(g_i) 个 context
  # 但每个 GPU 仍然只为自己的请求生成新 token

=== 逐层 Forward Pass ===
For each MoE layer j:
  [1] Attention (各 GPU 原地并行):
      token 在本地 GPU 用本地 context 执行 self-attention
      # 无需通信——context 已 coherent

  [2] Gating (各 GPU 并行, 共享参数):
      expert_idx = Top1_Gating(token_embedding)
      target_gpu = expert_placement[expert_idx][layer_j]

  [3] Single Alltoall Dispatch (仅 1 次):
      tokens 从当前 GPU 发送到持有 target expert 的 GPU
      # 关键差异: 无 Alltoall Combine

  [4] Expert FFN (目标 GPU 上原地):
      output = Expert_FFN(token)

=== Iteration 结束 ===
AllGather(new_tokens):
  每个 GPU 广播自己新生成的 token
  → 所有 GPU 更新 context 以保持 coherence
```

通信量对比（Table I）：
- baseline (DeepSpeed-MoE): Top-1 gating 下每层 2·G·N·L·p（dispatch + combine Alltoall）
- ExFlow (context-coherent): G·N·(L·p* + G)（dispatch Alltoall + iteration-end AllGather）
- 当 L（层数）较大时，iteration-end AllGather 开销被摊薄

术语一般如何实现？如何使用？
- **适用场景**：GPT-style auto-regressive 模型推理（decoder-only），token 生成后不再修改
- **不适用于 training**：training 时激活和梯度需要双向流动，context coherence 不适用
- **开销分析**：引入的 AllGather 开销 < 消除的 Alltoall Combine 开销，因为 AllGather 仅发生在 iteration 级别（每生成一个 token 一次），而非 per-layer 级别
- **与 affinity placement 协同**：context coherence 消除 50% Alltoall（combine），affinity placement 减少剩余 dispatch 的跨 GPU 比例

涉及论文标题：
- Exploiting Inter-Layer Expert Affinity for Accelerating Mixture-of-Experts Model Inference

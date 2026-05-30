## Expert-aware Batching for Pre-gated MoE（预门控MoE的专家感知批处理）

术语是什么？
Expert-aware Batching 是 Read-ME 基于 pre-gating router 提出的 MoE 推理批处理策略。传统 MoE 推理中 batch 内不同 token 可能选择不同 expert，导致每 batch 需激活大量 unique expert（Mixtral-8×7B 在 batch=56.8 时平均激活 7.63/8 experts），token 间需等待所有 expert 计算完成（implicit barrier），批处理效率退化至接近无批处理。Read-ME 利用 pre-gating 在推理前已知每个 token 的 expert 选择，将选择同一 expert 的 tokens 组 batch，保证 batch 内所有 token 共享相同 expert。Algorithm 1：维护 N 个 ReqQueueByExpert（每个 expert 一个 FIFO queue），每次从请求最多的 expert 队列取 tokens 填满 batch（MaxTokenLen），最大化 batch 内 expert 共享。因 Read-ME 的 expert 选择跨所有层一致（router 与 layer 无关），一个 batch 在所有层仅需加载相同 experts。实验将平均 unique experts/batch 从 5.08（decode-prioritized）降至 3.51。

从系统架构角度拆解术语：
调度流程（N=8 experts, MaxTokenLen=32）：

```
Initial:
  ReqQueue[0]=[A1,A2,A3,B1]    (4 tokens)
  ReqQueue[1]=[A4,A5,B2,B3,C1,C2] (6 tokens) ← 最多
  ReqQueue[2]=[B4,C3]           (2 tokens)
  ...

Iter1: E=expert1 (6 tokens), 6<32 → 取全部6 tokens
       ScheduledReq = [A4,A5,B2,B3,C1,C2]
Iter2: E=expert0 (4 tokens), 4+6=10<32 → 取全部4
       ScheduledReq = [A4,A5,B2,B3,C1,C2,A1,A2,A3,B1]
...

Result: 10 tokens 仅需2 unique experts (vs 传统5-8 unique experts)
```

术语一般如何实现？如何使用？
- 基于 DeepSpeed inference engine 修改 Scheduler：新增 ReqQueueByExpert（N 个 FIFO queue）。
- MaxTokenLen 由 GPU memory 可容纳的最大 token batch（hidden_size × seq_len + KV cache 容量决定）。
- 与 Prefill/Decode 兼容：算法按 expert 分组而非按 request phase，prefill 和 decode token 可混合。
- 延迟改善：端到端平均延迟 -5.0~6.1%，p95 延迟 -9.5~10.0%。

涉及论文标题：
- Read-ME: Refactorizing LLMs as Router-Decoupled Mixture of Experts with System Co-Design

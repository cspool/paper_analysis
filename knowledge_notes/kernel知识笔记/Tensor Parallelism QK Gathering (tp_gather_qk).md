## Tensor Parallelism QK Gathering (tp_gather_qk)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

tp_gather_qk 是 SPECPREFILL 在 vLLM + Tensor Parallelism 场景下实现的一个关键通信操作：在 speculator 完成 look-ahead decoding 后，由于各 TP rank 只持有 Query 和 Key 张量的一部分（按头数分片），需要跨 TP ranks 收集完整的 Q、K 张量以计算全局注意力矩阵，进而得到完整的 token 重要性分数。

从kernel调度角度拆解术语：

```
// TP=8 场景，每 rank 持有 H/8 个 head 的 Q,K
// Step 1: Speculator look-ahead (各 rank 独立)
for each rank r in TP_group:
    Q_r, K_r = speculator.forward(prompt_chunk)  // Q_r:[B, S, H/8, d]

// Step 2: tp_gather_qk
for each rank r:
    all_gather(Q_all, Q_r, group=TP_group)  // → [B, S, H, d]
    all_gather(K_all, K_r, group=TP_group)  // → [B, S, H, d]

// Step 3: Attention score computation (per-rank, full)
A_full = Q_all @ K_all^T  // [B, H, S, S] — 现在可以计算完整注意力
score = aggregate(A_full)  // max over H → [B, S]
```

术语一般如何实现？如何使用？

实现依赖 NCCL all_gather 在 TP group 内传输 Q、K 分片。由于仅在 speculator 完成 N 步 look-ahead 后调用一次，通信开销相对可控（Q、K 维度较小，speculator 为 8B 模型）。论文未给出具体 NCCL 调用细节，但概念上与标准 tensor parallelism 中的 all-reduce 操作类似，区别在于此处需要 gather（收集各 rank 分片）而非 reduce（求和）。

涉及论文标题：
- Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation

## Distributed Softmax (分布式 Softmax 聚合)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Distributed Softmax 是 Star Attention 阶段二中使用的跨 host 全局注意力聚合机制。各 host 独立对本地 KV cache 计算 local attention A_h = softmax(QK_h^T/√d) V_h 和 softmax denominator s_h = Σ exp(QK_{h,k}^T/√d)，然后将 (A_h, s_h) 发送到 query-host 进行全局聚合。Query-host 通过 online softmax（log-sum-exp trick）计算全局归一化的注意力输出。关键优势：通信开销与 context 长度 L 无关——每 token 仅传输 O(H·d) 数据（vector A_h + scalar s_h per host），而传统的 KV cache 传输需要 O(L·d)。

从系统架构角度拆解术语。

**Star Attention 阶段二的 Distributed Softmax 运转流程**（per decoder layer, per decode token）：

```
// H 个 hosts，h_q 为 query-host
// Step 1: 各 host 本地计算 (parallel)
for each host h:
    Q_h = project_q(current_token)        // [1, d_head] × n_heads
    A_h = FlashAttention(Q_h, K_h, V_h)   // local softmax attention
    s_h = Σ exp(Q_h @ K_h^T / √d)         // softmax denominator (one scalar per head per token)

// Step 2: Gather 通信 → h_q
gather {A_1, ..., A_H} → h_q              // H × d 的数据量
gather {s_1, ..., s_H} → h_q              // H × 1 的数据量

// Step 3: Online softmax log-sum-exp 聚合
s_global = s_1
A_global = A_1
for h = 2 to H:
    max_s = max(s_global, s_h)
    s_global = max_s + log(exp(s_global - max_s) + exp(s_h - max_s))
    A_global = exp(s_global_prev - s_global) * A_global + exp(s_h - s_global) * A_h

// Step 4: 仅在 h_q 更新 KV cache
```

**与 Ring Attention 通信对比**：
| 维度 | Ring Attention | Star Distributed Softmax |
|------|---------------|-------------------------|
| Prefill 阶段通信 | O(L·d) per layer, H 轮 ring exchange | 0（阶段一无跨 host 通信） |
| Decode 阶段通信 | O(L·d) per layer per token | O(H·d) per layer per token |
| 通信模式 | P2P ring (H sequential steps) | Gather to one host (1 step) |
| 随 context 长度扩展 | 线性增长（瓶颈） | 与 context 长度无关 |

术语一般如何实现？如何使用？

使用 Flash Attention 在每 host 上计算 local attention（利用 blockwise tiling 避免完整 attention matrix 物化），通过 NCCL all-gather 或 send/recv 完成 (A_h, s_h) 的收集。Online softmax 需在 query-host 上维护 running statistics（s_global, A_global），使用 FP32 精度避免数值溢出。Star Attention 同时支持 HuggingFace Transformers 和 TRT-LLM。实际使用中，多 hosts 通过 NCCL 通信，query-host 被动态指定为某一 host。

涉及论文标题：
- Star_Attention__Efficient_LLM_Inference_over_Long_Sequences

## Causal Attention / Masked Self-Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Causal Attention（因果注意力，也称为 Masked Self-Attention 或 Autoregressive Attention）是 decoder-only Transformer 的核心注意力机制，确保每个 token 只能 attend 到自身及之前的 token，不能看到未来 token。对于位置 t 的 token x_t，其查询向量 q_t 只能与位置 1..t 的键向量 k_i 和值向量 v_i 交互：

$$Attn(x_t; x_1, ..., x_{t-1}) = \sum_{i=1}^{t} softmax(\frac{q_t^T k_i}{\sqrt{d_k}}) v_i$$

这种因果约束是实现自回归语言建模（autoregressive language modeling）的关键——模型逐 token 预测下一个 token：P(x_t | x_1, ..., x_{t-1})。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Causal attention 的关键性质——token t 只依赖前 t-1 个 token 的 K/V——是 FOLDMOE 能实现 attention-MoE pipelining 的基础。具体来说：
- 将序列切分为微批次 X_{1:m}, X_{m+1:2m}, ...
- 计算 X_{m+1:2m} 的 attention 时，只需已缓存的 K_{1:m}, V_{1:m} 加上自身的 K_{m+1:2m}, V_{m+1:2m}
- 这使得 attention 层可以沿 sequence 维度流水线化，在计算后续微批次的同时，前序微批次已可进入 MoE 层的 A2A 通信

```
# 微批次间的 KV 累积
for mb in micro_batches:
    K_mb, V_mb = proj_kv(X_mb)
    K_cache = concat(K_cache, K_mb)   # 逐步累积
    V_cache = concat(V_cache, V_mb)
    Z_mb = attention(Q_mb, K_cache, V_cache, causal_mask)  # 只 attend 到前缀
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Causal attention 的标准实现方式：
1. **Naive**: 计算完整 N×N attention 矩阵后应用上三角 mask（设为 -∞），O(n²) 内存
2. **FlashAttention**: fused kernel，分 tile 计算，IO-aware，将 softmax 在线计算融入 tile 循环，避免物化完整 attention 矩阵
3. **PagedAttention (vLLM)**: 用于推理的 KV cache 管理，将 KV cache 分页存储

在 FOLDMOE 中，使用 FlashAttention 作为 attention 实现，每个 micro-batch 内的 causal attention 计算与原全序列 causal attention 产生相同的输出（因 mask pattern 一致）。

涉及论文标题：
- FOLDMOE: Efficient Long Sequence MoE Training via Attention-MoE Pipelining

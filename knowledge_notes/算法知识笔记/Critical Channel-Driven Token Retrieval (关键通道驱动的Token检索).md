## Critical Channel-Driven Token Retrieval (关键通道驱动的Token检索)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Critical Channel-Driven Token Retrieval 是 TailorKV 提出的动态 token 选择技术，利用 query 和 key 中 outlier channel（值显著大于其他 channel 的 hidden dimension）来近似 attention score，从而精准选择需要从 CPU 取回 GPU 的 Top-K 个 KV token。核心洞察：(1) attention score = q·K^T 中，每个 channel 对 attention 的贡献为 |q_i| · |K_i|；(2) 少数 channel 在 query 和 key 中呈现大幅度值（outlier）、主导 attention 计算；(3) 仅用这些 critical channels 的 query/key 子集计算近似 attention，便可高精度识别最重要的 token。通道数 d_s = 8（LongBench）/ 12（InfiniteBench/RULER），覆盖的 token 检索准确率接近完整 attention 检索。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// === Decoding 阶段，sparsity-friendly 层 l ===

// Stage 1（在 layer l-1 执行，用于预取）:
h_{l-1} = hidden_state_after_layer_l_minus_1        // shape: (1, d)
q_hat = W_q[l] @ h_{l-1}                             // inter-layer query 预估

// 计算 channel 重要性
for i in 1..d_h:
    s_i = |q_hat_i| * max(|K_{CPU}[i, :]|)           // Eq.(10)

critical_ch = Top_indices(s, d_s)                     // 选 d_s 个 critical channels
Prefetch_async(K_{CPU}[critical_ch, :])               // 从 CPU 异步预取 critical key

// Stage 2（在 layer l 执行）:
q = W_q[l] @ h_l                                     // 真实 query
q_crit = q[critical_ch]                              // d_s 维
K_crit = K_prefetched[critical_ch, :]                // (d_s, n)，已预取完成

// 近似 attention scores（用 d_s 而非 d_h 维计算）
a_approx = q_crit @ K_crit.T / sqrt(d_s)             // shape: (1, n)
topk_idx = TopK(a_approx, k=n_topk)                   // 选 Top-K

// 从 CPU 取完整 KV
Fetch_sync(K_full[topk_idx], V_full[topk_idx])        // 唯一不可 overlap 的操作

// 合并 local + fetched tokens 做完整 FlashAttention
output = FlashAttn(q, cat(K_local, K_fetched), cat(V_local, V_fetched))
```

为什么 dynamic channel selection 优于 static：TailorKV 实验（Figure 9b）显示 query/key 的 outlier 位置不是固定的，它们可能出现在任何 channel（Figure 2 Bottom），因此离线标定的 static channel set 召回率低于运行时动态选择。

术语一般如何实现？如何使用？

实现要点：(1) `max(|K_i|)` 在 prefill 后计算一次并存储在 CPU 元数据中（不随 decoding 变化），仅在每次有新 token 加入时更新；(2) Stage 1 的 q_hat 利用 inter-layer 相似性（余弦相似度 >0.99 between adjacent hidden states, Appendix B Figure 11）提前一层预估，使 critical key 预取可以与 layer l-1 的计算重叠；(3) Stage 2 的 attention 近似使用 d_s 维，计算量仅为完整 attention 的 d_s/d_h（约 8/128 = 6.25%）；(4) Double buffering——GPU 上有两个 buffer（读/写），一边写入 layer l 的 prefetch 数据、一边读取 layer l-1 预取好的数据。

该技术与 ANN-based token retrieval（如 Faiss/LSH）的区别：不需要额外 CPU 端索引构建和检索计算，所有选择逻辑在 GPU 上完成（仅 critical channel 的 K 从 CPU→GPU，然后 GPU 上近似 attention+排序），避免了 PQCache/MagicPIG 的 CPU 端 K-Means clustering 或 LSH hashing 开销。

涉及论文标题：
- TailorKV: A Hybrid Framework for Long-Context Inference via Tailored KV Cache Optimization

---

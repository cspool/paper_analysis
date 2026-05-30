## Attention-Free KV Cache Eviction (注意力无关的 KV 缓存逐出)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Attention-Free KV Cache Eviction 是一类不依赖 attention maps（softmax(QK^T) 输出）来识别重要 token 的 KV cache 压缩方法。与 Attention-Based 方法（H2O、SnapKV、TOVA 等需 prefill 阶段完整 attention scores）不同，Attention-Free 方法仅使用 token 位置或跨层结构来决定保留哪些 KV pairs。代表方法：StreamingLLM（attention sink + sliding window）、LaCache（ladder-shaped 跨层位置模式）。

核心优势：与 FlashAttention 完全兼容。FlashAttention 的 IO-aware tiling + online softmax 不物化完整 S ∈ R^{n×n}，因此 Attention-Based 方法要么放弃 FlashAttention（降速），要么在内核中额外输出 scores（增加 overhead）。Attention-Free 方法完全规避此问题。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Attention-Free (LaCache/StreamingLLM) — 无需 S 矩阵
for layer l in 1..L:
    keep_range = compute_range_from_position(l, S, O)
    K_cache[l] = K[l, keep_range]     # 仅依赖 token 位置
    V_cache[l] = V[l, keep_range]
# FlashAttention 正常加载压缩后的 KV tiles

# Attention-Based (H2O/SnapKV) — 需要完整 S 矩阵
S = Softmax(Q @ K^T / sqrt(d))        # 需要 materialize S
importance = aggregate(S)
keep_idx = topk(importance, budget)
# 问题：FlashAttention 不产出 S
```

术语一般如何实现？如何使用？

即插即用集成到 HuggingFace Transformers attention 层，无训练/模型修改，计算 overhead 极低（仅 tensor indexing）。LaCache 在 H200 上实现 score-throughput Pareto 最优 (LongBench)，超越所有 Attention-Based baselines。代码：StreamingLLM https://github.com/mit-han-lab/streaming-llm，LaCache https://github.com/GATECH-EIC/LaCache。

涉及论文标题：
- LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models
- Efficient Streaming Language Models with Attention Sinks (StreamingLLM)
- LagKV: Lag-Relative Information of the KV Cache Tells Which Tokens Are Important

---

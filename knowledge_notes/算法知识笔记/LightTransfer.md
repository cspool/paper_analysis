## LightTransfer

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

LightTransfer 是将预训练 Transformer 转换为 Hybrid 模型（full attention + streaming attention 层混合）的轻量框架（arxiv 2410.13846）。核心洞察：利用 LLM 不同层在长上下文推理中的功能差异——某些层是"懒惰层"（注意力集中在 sink + recent tokens），将其 full attention 替换为 streaming attention 以降低 KV cache 显存；非懒惰层保留 full attention 维持全局信息捕获。

两种模式：(1) LightTransfer-TEST：test-time 在线转换，prefilling 阶段利用 FlashAttention LSE 值计算 lazy ratio，优先队列动态识别懒惰层——无需任何训练；(2) LightTransfer-TRAIN：训练集统计懒惰层频率预选，然后 SFT 微调（~5K 样本，原用于 long-reasoning 蒸馏的数据）——适用于短输入长推理。

理论保证：Theorem 5.1 证明输出误差 ≤ 被移除 KV 对的注意力分数之和 × 常数，而 lazy ratio 算法恰好优化该上界的 greedy 版本。

从算法pipeline角度拆解术语。

**LightTransfer-TEST 算法**：
```
Q = PriorityQueue(maxsize = P_ratio * L)  # max-heap, key = lazy_ratio
for i in 0..L-1:
    O_i, lse_i = FlashAttention(LN(X_{i-1}), causal=True, return_lse=True)
    r_i = compute_lazy_ratio(lse_i, ...)
    Q.push((r_i, i))
    if Q.is_full():
        r_max, lazy = Q.pop()
        K_cache[lazy] = K_cache[lazy][:w_sink] + K_cache[lazy][-w_recent:]  # 缩减
        V_cache[lazy] = V_cache[lazy][:w_sink] + V_cache[lazy][-w_recent:]

# Decoding: 使用已缩减的 KV cache
```

术语一般如何实现？如何使用？

基于 PyTorch + HuggingFace Transformers + FlashAttention。开源：https://github.com/sail-sg/LightTrans，HuggingFace 模型：cxdu/QwQ-32B-LightTransfer。超参数：$w_{\text{sink}}=4$, $w_{\text{recent}}=1020$, $w_{\text{last}}=32$, 标准层保留比例 50%-75%。结果：50% 层替换时吞吐 2.17× (16K seqlen)，LongBench 下降 <1.5%，AIME24 达 53.3%（QwQ-STILL baseline 46.7%：+6.6%）。

涉及论文标题：
- LightTransfer: Your Long-Context LLM is Secretly a Hybrid Model with Effortless Adaptation

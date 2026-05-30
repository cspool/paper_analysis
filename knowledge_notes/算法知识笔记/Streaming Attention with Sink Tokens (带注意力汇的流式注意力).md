## Streaming Attention with Sink Tokens (带注意力汇的流式注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Streaming Attention 是 StreamingLLM（Xiao et al., ICLR 2024）提出的注意力机制：将 full causal attention 替换为固定大小的注意力窗口，仅保留 (1) 前 $w_{\text{sink}}$ 个 attention sink token（因 softmax 归一化累积了大量注意力权重），(2) 最近 $w_{\text{recent}}$ 个 token（局部上下文窗口）。KV cache 大小从 $O(L)$（随序列长度线性增长）降为 $O(w_{\text{sink}} + w_{\text{recent}})$ = 常数，支持无限长上下文流式推理。

与 "Streaming Attention Heads"（DuoAttention/PruLong 的 head 级分类）的区别：streaming attention 是整层粒度的 attention 机制替换而非逐 head 分类。

从算法pipeline角度拆解术语。

```
# Full attention（第 n 步）: Q_n=[1,d], K_cache=[n,d], V_cache=[n,d]
S = Q_n @ K_cache^T / sqrt(d)    # [1, n] — 随 n 增长

# Streaming attention（固定 KV cache 大小）
K_stream = concat([K_cache[:w_sink], K_cache[-w_recent:]])  # [w_sink+w_recent, d]
V_stream = concat([V_cache[:w_sink], V_cache[-w_recent:]])
S_stream = Q_n @ K_stream^T / sqrt(d)  # [1, w_sink+w_recent] — 常数
A_stream = softmax(S_stream)
O_stream = A_stream @ V_stream
```

术语一般如何实现？如何使用？

StreamingLLM 将所有层替换为 streaming attention，典型配置 $w_{\text{sink}}=4$, $w_{\text{recent}}=1020$，窗口 ≈ 1024 tokens。全部替换导致 LongBench 平均下降 3.5-11.5%（全局信息捕获能力完全移除）。LightTransfer 的创新：仅在"懒惰层"使用 streaming attention，非懒惰层保留 full attention 作为全局信息锚点——50% 层替换时吞吐提升 2.17×，LongBench 仅下降 <1.5%。开源：https://github.com/mit-han-lab/streaming-llm。

涉及论文标题：
- LightTransfer: Your Long-Context LLM is Secretly a Hybrid Model with Effortless Adaptation
- StreamingLLM: Efficient Streaming Language Models with Attention Sinks

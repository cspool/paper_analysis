## Attention Sink

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Attention Sink 是 StreamingLLM（Xiao et al., ICLR 2024）发现的 LLM 注意力现象：首 token（bos token 或序列第一个 token）在所有 generation step 中持续获得异常高的 Attention Score，即使该 token 语义上并不重要。其根本原因在于 Softmax 归一化必须让概率之和为 1——深层 head 中大量 token 的注意力分数趋近于 0，模型将"多余的注意力质量"倾泻到首 token 这个始终可用的"接收槽"上。

在 KV Cache 压缩中，Attention Sink 具有双重意义：(1) 必须保留首 token，否则准确率急剧下降；(2) 好的 token 剪枝方法应当自然地保留 Attention Sink token。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Attention Sink 在 token 剪枝中的角色**：

```
// StreamingLLM 策略（固定框架）
cache = [sink_tokens] + [recent_tokens]  // 始终保留前 4 个 token + 最近 W 个 token

// A2SF 中 Attention Sink 被自然保留
// 即使 α 衰减历史分数，sink token 每步都获得极大 Score
// Σ α^{n-q} × S_q,0 的衰减被连续高分抵消 → sink token 保持高分
```

**A2SF 论文验证（Section 4.3）**：
A2SF 即使施加 Forgetting Factor，Sink Token 仍然被选中且保持高分——"这是因为即使施加遗忘因子，Sink Token 每步都输出较大的值，相比其他 token 保持较高值"。

术语一般如何实现？如何使用？

在 KV Cache 管理策略中，最简单的是始终将首 token 加入"不可逐出"列表。StreamingLLM 固定保留前 4 个 token。H2O 和 A2SF 中，由于首 token 的 A2S/A2SF 分数天然很高，通常无需特殊处理即被自然保留。这已被 A2SF 的实验验证（A2SF 下 Attention Sink token 被正常选中）。

在 HISA 中，Attention Sink 现象被用于层级索引的 block 选择策略：HISA 的 block-level 粗过滤中，**首 block 和尾 block 被强制包含**在候选集 C_t 中（C_t = TopK(J_{t,:}, m) ∪ {first block, last block}）。原因是首 block 包含 attention sink tokens（模型将"多余注意力"倾泻至此），尾 block 包含局部上下文（query 自身附近的 tokens 通常有高 attention）。这一强制包含确保 HISA 的粗过滤不会因 block-level 近似而丢失这两类关键信息。同样的策略也见于 MoBA (Lu et al., 2025)。

涉及论文标题：
- A2SF: Accumulative Attention Scoring with Forgetting Factor for Token Pruning in Transformer Decoder
- DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads
- Cache Me If You Can: How Many KVs Do You Need for Effective Long-Context LMs
- HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention
- InfiniteHiP: Extending Language Model Context Up to 3 Million Tokens on a Single GPU
- LogQuant: Log-Distributed 2-Bit Quantization of KV Cache with Superior Accuracy Preservation
- LagKV: Lag-Relative Information of the KV Cache Tells Which Tokens Are Important
- MagicPIG: LSH Sampling for Efficient LLM Generation
- PowerAttention: Exponentially Scaling of Receptive Fields for Effective Sparse Attention
- WindowKV: Task-Adaptive Group-Wise KV Cache Window Selection for Efficient LLM Inference
- Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression
- Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation
- Star_Attention__Efficient_LLM_Inference_over_Long_Sequences (Star Attention leverages attention sink phenomenon: anchor blocks shift block-local attention sinks to a single sink, enabling block-local attention to approximate global attention)
- ZSMerge: Zero-Shot KV Cache Compression for Memory-Efficient Long-Context LLMs
- TokenButler: Token Importance is Predictable

**Q-Filters 论文中的 Attention Sink 处理**：Q-Filters 在 NIAH 实验中不对前两层进行 KV Cache 压缩（"we do not compress key-value pairs in the first two layers of the models"），这与 Attention Sink 通常在浅层表现更显著的现象一致。由于 Q-Filters 通过 Key 在 Query 主方向上的投影来估计重要性，而 Attention Sink token 的 Key 通常在该方向上有显著投影（因其高注意力），因此 Q-Filters 自然倾向于保留 Attention Sink token。

**MagicPIG贡献的几何解释**（Section 3）：MagicPIG对Attention Sink提供了三个关键几何发现——(1) k_sink（首token的key state）在不同输入token下朝向几乎不变（相似度>0.99）；(2) k_avg（所有key的均值向量）在不同输入句下朝向稳定（相似度>0.9）；(3) k_sink和k_avg几乎相反（余弦相似度-0.9~-0.8）。这三者形成了Figure 2c的几何：query接近k_0方向，key集中在相反方向的窄锥中（除sink token外）。该几何解释了为什么TopK搜索困难（q和k分布方向相反→NN搜索效果差），以及为什么LSH需要centering（否则几乎所有key与q的碰撞概率接近0）。

**SPECPREFILL 中的 Attention Sink 缓解**：SPECPREFILL 识别到 Attention Sink 现象会扭曲 token 重要性估计（首几个 token 倾向于获得过高的注意力权重），通过 **Look-ahead Decoding**（向前解码 N=8 步，聚合 N 个解码 token 的注意力而非仅依赖最后一个 token 的注意力）来缓解这一偏差。这种策略与仅使用最后 token 注意力的方法相比，显著减少了对 sink token 的过拟合选择。

---

## StreamingLLM (流式LLM注意力机制)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

StreamingLLM（Xiao et al., 2024）是一种免训练的 KV cache 压缩方法，核心发现是 LLM 的注意力矩阵中存在 **attention sink** 现象——初始若干个 token 会持续获得不成比例的高注意力分数（即使它们在语义上不重要）。基于此提出：在 decoding 阶段，KV cache 仅保留 attention sink（初始 4 个 token）+ 最近局部窗口（W 个 token），丢弃中间所有 token 的 KV。

在 MInference 论文中，StreamingLLM 被用作 baseline（对应 A-shape 模式），参数为 1K global tokens + 4K local window。在 pre-filling 阶段的评测中，StreamingLLM 在 retrieval 类任务上表现极差（InfiniBench Retr.KV: 0.8 vs Full Attention 14.4, RULER 有效 context 仅 4K vs Full 16K），因为一旦关键信息超出 local window 范围，模型无法访问。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# StreamingLLM 的 KV cache 管理（decoding 阶段）
GLOBAL = 4    # attention sink tokens
LOCAL = W     # 局部窗口大小

# 每次 decode step:
KV_cache = KV_cache[:GLOBAL] + KV_cache[-LOCAL:]  # 丢弃中间 token
# 新 token 的 KV 始终追加到局部窗口末尾

# 注意力计算（所有 head 统一）
M[i, j] = 1 if (j < GLOBAL) or (j >= i - LOCAL) else 0
A = softmax(Q @ K^T / √d - c * (1 - M))
```

在 MInference 的 pre-filling 场景中：
- Global tokens 扩展到 1K（而非 4），因为 pre-filling 需要更多 global context
- Local window 设为 4K
- Decoding 阶段保持 dense 计算（不做稀疏）

术语一般如何实现？如何使用？

StreamingLLM 实现简单——在 PyTorch 中可通过修改 attention mask 实现：
```python
mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
mask[:, GLOBAL:-LOCAL] = False  # 屏蔽中间 token
```

使用场景：适合对局部上下文依赖强的任务（如 language modeling、对话），但不适合 retrieval、multi-hop QA 等需要全局上下文的任务。在 MInference 的分类中，StreamingLLM 等同于 "Ours w/ only A-shape"，代表了仅依赖静态局部模式的稀疏注意力方法的性能上限。

核心局限：无法处理需要非局部信息的任务（KV retrieval 准确率接近 0），因为关键信息的 token 可能位于 global window 和 local window 之间。

涉及论文标题：
- MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention
- MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding
- Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving

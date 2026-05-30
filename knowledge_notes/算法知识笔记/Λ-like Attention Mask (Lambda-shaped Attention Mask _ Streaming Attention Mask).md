## Λ-like Attention Mask (Lambda-shaped Attention Mask / Streaming Attention Mask)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Λ-like Attention Mask 是一种特殊的因果注意力遮罩模式，最早由 LM-Infinite（Han et al., 2023）和 StreamingLLM（Xiao et al., 2023b）提出，因形状类似希腊字母 Λ 而得名。该 mask 使每个 query token 仅 attend 到：(1) 序列开头的固定数量 token（attention sinks，构成 Λ 的左分支）；(2) 自身之前的最近 W 个 token（sliding window，构成 Λ 的右分支）。序列中间的 token 被完全跳过（mask 置为 -∞），不参与 attention 计算。

数学表达：M_streaming[i,j] = 0 if (j ≤ S) or (i - j ≤ W and j ≤ i) else -∞，其中 S 为 sink token 数量，W 为 recent window 大小。

从算法pipeline角度拆解术语。

```
# Λ-like Mask 的 attention 计算
def streaming_attention(Q, K, V, S, W):
    """
    Q, K, V: [batch, heads, seq_len, dim]
    S: number of attention sinks (initial tokens)
    W: recent window size
    """
    seq_len = Q.shape[2]

    # 构建 Λ-like mask
    mask = torch.full((seq_len, seq_len), float('-inf'))

    # 每个 query i 可以 attend 到:
    for i in range(seq_len):
        # (1) attention sinks: 前 S 个 token
        mask[i, :min(S, i+1)] = 0
        # (2) recent window: 最近 W 个 token
        start = max(0, i - W + 1)
        mask[i, start:i+1] = 0

    scores = (Q @ K.transpose(-1, -2)) / sqrt(d)
    scores = scores + mask  # -∞ → softmax 后概率为 0
    attn_weights = softmax(scores, dim=-1)
    return attn_weights @ V
```

**在 DuoAttention 中的应用**：Λ-like mask 是 streaming heads 的 attention 计算核心，替代 causal mask。在 deployment 中 streaming heads 仅使用该 mask 计算 attention：attn_streaming = softmax(QK^T ⊙ M_streaming)V。DuoAttention 使用 S=64（sink tokens）和 W=256（recent tokens），KV cache 保持 constant O(S+W) 大小。在 chunked pre-filling 中，每个 chunk 的 KV 计算完毕后立即 prune streaming heads 的 KV cache（仅保留 sink + recent），下一 chunk 的 attention 仅需处理 constant 数量历史 token。

术语一般如何实现？如何使用？

StreamingLLM 首次系统化使用该 mask 实现 infinite-length 流式推理（无需重新预训练即可处理超预训练长度的输入）。DuoAttention 将其限定在 streaming heads 类别上使用（retrieval heads 使用完整 causal mask）。实现方式：修改标准 FlashAttention kernel 的 mask 输入（将 Λ-like mask 作为 block-wise attention mask），或使用 FlashInfer 的 block-sparse attention 模块（Guo et al., 2024）。DuoAttention 训练阶段用 block-sparse approximation 加速 Λ-like attention 计算。

涉及论文标题：
- DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Forgetting Factor α（0 < α < 1）是 A2SF 方法的核心创新。它是一个指数衰减系数，在累积 Attention Score 时反复乘以历史分数，使得越早产生的 Attention Score 收敛至 0，消除 Causal Mask 导致的 token 位置偏差。

公式：$A_{n,k}^h = \sum_{q=1}^n \alpha^{n-q} \times S_{q,k}^h$

展开形式：$A_{n,k}^{h} = S_{n,k}^{h} + \alpha \cdot S_{n-1,k}^{h} + \alpha^{2} \cdot S_{n-2,k}^{h} + \dots + \alpha^{N-k} \cdot S_{k,k}^{h}$

α 的含义：
- α = 1.0：等价于 H2O 的原始 A2S，无衰减，全量历史累积
- α = 0.0：完全忽略历史，仅用当前步 Attention Score 决定重要性
- α ∈ [0.1, 0.3]：论文实验发现的最优范围，主要考虑近期历史
- α → 0：快速收敛，仅看近期趋势——适合区分度高的数据集
- α → 1：缓慢收敛，长历史仍影响——适合需记忆早期关键信息的数据集

该设计受人类遗忘曲线（Ebbinghaus Forgetting Curve）启发——其简化形式为指数型。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**A2SF 伪代码**：

```
alpha = 0.2  // Forgetting Factor, [0.1, 0.3] 为最优范围

// 初始化
A = zeros(N)  // 每个 token 的 A2SF 分数

for n in 1..max_gen:
    // Step 1: 衰减所有已有分数
    A *= alpha  // 所有历史分数 × α（等效于 α^{new_q - old_q}）

    // Step 2: 加入当前步的 Attention Score
    S_n = softmax(Q_n @ K^T / sqrt(d_k))  // [1, n]
    for k in 1..n:
        A[k] += S_n[k]  // 注意：S_n[k] 是 q=n, key=k 的分数

    // Step 3: 按 A2SF 分数选择保留 token
    keep_indices = top_k(A, K)
    evict_indices = rest

    // A2SF 不分配 local cache，全量 budget 用于 selective
```

**关键运算细节**：
- 每步先对全局 A 做 `A *= alpha`，一次乘法即实现所有历史分数多一次衰减
- 再累加当前步分数：`A[1:n] += S_n[1:n]`
- 时间复杂度 O(n)，与 H2O 的 O(n) 相同，无额外开销

术语一般如何实现？如何使用？

A2SF 以即插即用方式集成到 HuggingFace Transformers 推理流程。每次 Attention 计算后，调用 `cache.evict_by_a2sf(k, alpha)` 完成选择。用户只需设置 α 和 cache_ratio 两个参数。代码开源：https://github.com/Dirac-Notation/A2SF。A2SF 可与后续处理不重要 token 的技术（No Token Left Behind 的量化、Get More with LESS 的低秩分解、Keyformer 的 Gumbel-Softmax）兼容叠加。

涉及论文标题：
- A2SF: Accumulative Attention Scoring with Forgetting Factor for Token Pruning in Transformer Decoder

---

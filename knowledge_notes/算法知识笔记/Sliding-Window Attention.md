## Sliding-Window Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sliding-Window Attention（滑动窗口注意力）是 Child et al. (2019) 在 Sparse Transformers 中提出的稀疏注意力机制。与标准 self-attention 中每个 query 关注所有历史 token 不同，sliding-window attention 限制每个 query 仅关注前 C 个 token（窗口大小 C 为常量），使用 window causal mask B：`B_{ij}=0 if i-C<j≤i else -∞`。这使得 KV cache 内存复杂度从 O(N) 降至 O(C)，即内存使用量是常量，不随序列长度增长。YOCO 将其作为 Self-Decoder 的备选高效自注意力模块（与 gated retention 并列），利用其 O(1) 推理内存的特性来降低整体 KV cache 开销。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Sliding-window attention 在 YOCO Self-Decoder 中的计算过程：

```python
# Sliding-Window Self-Attention (YOCO Self-Decoder)
# Input: X ∈ R^{N×d}, window_size C (e.g., 1024)
# Weights: W_Q, W_K, W_V, W_O ∈ R^{d×d}

Q, K, V = X@W_Q, X@W_K, X@W_V  # shape: [N, d]

# Window causal mask: each query i attends to keys in [i-C+1, i]
B = zeros(N, N)
for i in range(N):
    for j in range(N):
        if i - C < j <= i:       # within window
            B[i,j] = 0
        else:                     # outside window
            B[i,j] = -inf

# Multi-head computation
for h in range(num_heads):
    scores = Q_h @ K_h^T / sqrt(d_head)   # [N, N]
    scores = softmax(scores + B)           # masked softmax
    head_h = scores @ V_h                  # [N, d_head]

# Output projection
Y = concat(head_1, ..., head_H) @ W_O
```

**Annotations**: 推理时仅缓存每个 head 的最近 C 个 token 的 K, V（而非全部 N），cache size = C × d_head × H_kv × layers。对于 YOCO Self-Decoder，C=1024。Prefill 阶段 window attention 的计算复杂度为 O(N×C×d)，而非 O(N²×d)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Sliding-window attention 常用于：(1) 长文档建模——通过限制 attention range 获得线性级别的计算复杂度；(2) 作为高效 attention 模块嵌入混合架构——如 YOCO 的 Self-Decoder 或 Jamba 的 hybrid Mamba-Transformer；(3) 推理优化——固定窗口使得 KV cache 大小与序列长度解耦，适合流式/实时应用。实现时可使用 FlashAttention 的 windowed attention kernel 或自定义 Triton kernel。主要限制：窗口外信息完全丢失（缺乏全局 attention），在 YOCO 中由 Cross-Decoder 的全局 cross-attention 弥补；对于需要全局上下文的任务单独使用时可能精度不足。

涉及论文标题：
- Efficient implementations for emerging model architectures (YOCO: You Only Cache Once)
- Hymba: A Hybrid-head Architecture for Small Language Models

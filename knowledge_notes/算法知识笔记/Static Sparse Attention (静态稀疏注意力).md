## Static Sparse Attention (静态稀疏注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

静态稀疏注意力（Static Sparse Attention）是一种在推理全过程中使用预定义的、固定的注意力掩码模式的稀疏注意力机制。与动态稀疏注意力不同，静态模式不根据输入内容在线调整注意力掩码，而是在推理前就已确定每个 token 可关注哪些位置的 token。PowerAttention 论文将静态稀疏注意力的设计问题形式化为：在 DAG（有向无环图）中找到最优边集，使得在固定出度约束（sparsity constraint）下，多步可达节点数最大化。

常见的静态稀疏注意力模式包括：(1) Sliding Window——每个 token 仅关注前 W 个 token；(2) Stride Slash——在 sliding window 基础上按等间距放置 slash token；(3) Dilated Attention——使用膨胀滑动窗口（如每隔一个位置跳过）；(4) LongNet——多 mask 叠加，以几何增长的 segment length 和 dilation ratio 构建；(5) PowerAttention——每个 token 关注距离为 2 的幂次的位置。静态模式的共同优势：训练阶段可进行效率优化（mask 固定可预编译），解码阶段对新 token 处理更高效（无需重新计算 mask），且实现简洁。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**静态稀疏注意力的 DAG 建模（PowerAttention 论文 Section 3.1）**：

静态稀疏注意力掩码可以自然地解释为 DAG 的邻接矩阵。对 d 层 LLM：
- 节点：每个 token 在特定位置
- 边：token i 可以 attend token j（j < i，因果性）
- 单层感受野（out-degree）：token i 直接关注的所有 token 集合 A_i
- 多层感受野（d-step reachability）：经过 d 层信息传播后，token i 能间接访问的所有 token

```
# Static Sparse Attention 的一般形式（伪代码）
# q_idx [M, 1], kv_idx [1, N]
block_size = 256  # CUDA block size

# 1. Sink tokens（序列开头的注意力汇标记）
mask_sink = kv_idx < block_size

# 2. Local window（局部上下文窗口）
blk_qk = q_idx // block_size - kv_idx // block_size
mask_window = blk_qk < window_size

# 3. Pattern-specific mask（各静态模式的核心差异）
# Sliding Window: 无额外 mask
# Stride Slash: mask_slash = blk_qk % stride_size == 0
# Dilated: mask_dilated = (blk_qk & 1 == 0) & (blk_qk < window_size)
# PowerAttention: mask_power = (blk_qk & (blk_qk - 1)) == 0

# 4. 因果性 + 组合
causal = q_idx >= kv_idx
mask = causal & (mask_window | mask_pattern | mask_sink)
```

**DAG 可达性分析**（各静态模式的路径复杂度，到达距离 N 的 token）：

$$
\begin{aligned}
\text{Sliding Window:} &\quad O(N) \text{ layers} \quad \text{(线性扩展)} \\
\text{Stride Slash:} &\quad O(\sqrt{N}) \text{ layers} \quad \text{(平方根扩展，有覆盖间隙)} \\
\text{Dilated:} &\quad O(N) \text{ layers, } \sim 50\% \text{ 覆盖率} \quad \text{(奇数距离不可达)} \\
\text{LongNet:} &\quad O(\log N) \text{ layers} \quad \text{(有覆盖盲区)} \\
\text{PowerAttention:} &\quad O(\log N) \text{ layers, } 100\% \text{ 覆盖率} \quad \text{(指数扩展)}
\end{aligned}
$$

术语一般如何实现？如何使用？

静态稀疏注意力通常通过定义固定的 attention mask 实现。在 PyTorch 中可直接使用 `torch.nn.functional.scaled_dot_product_attention` 的 `attn_mask` 参数，或使用 FlexAttention 的 `score_mod` 函数定义 mask。实现时通常采用 block-sparse 策略（block_size=64~256 tokens）以对齐 GPU memory access 模式。静态模式特别适用于：(1) 需要训练阶段效率优化的场景——mask 预编译为 block-sparse kernel；(2) 高稀疏度场景（>90% sparsity）——避免动态估计开销大于稀疏计算收益；(3) streaming/continuous batching 场景——新 token 无需重新估计 mask。

PowerAttention 论文的关键发现：尽管相同稀疏度下各静态模式的单层感受野大小相同（out-degree 相同），但多层信息传播后可达节点数差异巨大——设计良好的模式（如 PowerAttention）在 6 层后可覆盖全部 32K token，而 Sliding Window 仅覆盖约 2304×6≈14K token 的范围（且最后 token 无法访问序列开头的 token）。

涉及论文标题：
- PowerAttention: Exponentially Scaling of Receptive Fields for Effective Sparse Attention

---

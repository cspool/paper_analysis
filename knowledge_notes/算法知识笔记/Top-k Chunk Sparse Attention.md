## Top-k Chunk Sparse Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Top-k Chunk Sparse Attention 是 RWKV-X 论文中提出的一种稀疏注意力机制，灵感来自 MoBA（Mixture of Block Attention, Lu et al., 2025）。核心思想是将自回归生成中的全序列注意力替换为仅对 top-k 个最相关 chunk 的稀疏注意力，从而将 O(N²) 的复杂度降至 O(kBN) ≈ O(N)。具体流程：(1) 将长度为 N 的输入序列等分为 n 个大小为 B 的 chunk；(2) 对每个 query token q，计算 q 与各 chunk 的 mean-pooled key vector 的内积作为相关性得分 s_i = q · (1/B Σ_j k_j^(i))；(3) 通过 TopK 操作选择得分最高的 k 个 chunk 索引 I = TopK({s_i}, k)；(4) 仅在被选中的 chunk 上计算标准 softmax attention: Attn(q, K_I, V_I) = softmax(qK_I^T/√d_k) V_I。由于 k 和 B 为小常数，总计算复杂度为 O(kBN) ≈ O(N)。该方法结合了 KV Cache Management（SnapKV 风格的重要性逐出）以确保解码阶段的 constant memory usage。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Top-k Chunk Sparse Attention 在 RWKV-X 中的前向流程：
```
# Input: h ∈ R^{B×L×D}, chunk_size B_c, topk k
q, k, v = W_Q(h), W_K(h), W_V(h)  # (B, L, d_head)

# Step 1: Chunk partitioning
n_chunks = L // B_c
k_chunks = reshape(k, (B, n_chunks, B_c, d_head))  # (B, n, B_c, d)
v_chunks = reshape(v, (B, n_chunks, B_c, d_head))

# Step 2: Mean-pooled chunk keys
k_mean = mean(k_chunks, dim=2)  # (B, n, d)

# Step 3: Chunk relevance scoring
scores = einsum("bld,bnd->bln", q, k_mean)  # (B, L, n)

# Step 4: Top-k chunk selection
topk_indices = topk(scores, k, dim=-1)  # (B, L, k)

# Step 5: Gather selected chunks
k_selected = gather(k_chunks, topk_indices)  # (B, L, k*B_c, d)
v_selected = gather(v_chunks, topk_indices)

# Step 6: Sparse attention
attn = softmax(q @ k_selected^T / sqrt(d_k))  # (B, L, k*B_c)
output = attn @ v_selected  # (B, L, d)

# Step 7: Output projection + residual
h_out = h + W_O(output)
```

在 RWKV-X 混合架构中，稀疏注意力层占约 25%（12 层中约 3 层），其余 75% 仍为 RWKV-7 循环层。在解码阶段，KV cache 通过 SnapKV 风格的重要性管理保持固定大小（论文中设为 64K），使 attention 计算量和内存均不随生成长度增长。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/howard-hou/RWKV-X。RWKV-X 的实现将稀疏注意力层与 RWKV-7 层混合堆叠，约每 4 层插入 1 个稀疏注意力层（通过消融实验确定 25% 注意力比例最优，Figure 5）。Block expansion 阶段零初始化新稀疏注意力层（output projection=0），确保初始状态下新层表现为恒等映射，仅传递残差。Alignment pretraining 阶段仅训练稀疏注意力层参数（freeze RWKV-7 参数），Long-context pretraining 阶段全参数微调。适用于需要长上下文检索能力但希望保持线性复杂度的 LLM 训练和推理。

与 MoBA 的区别：
- MoBA 使用 parameter-less gating 选择 top-k block，每个 query 独立选择 block
- Top-k Chunk Sparse Attention 同样使用 mean-pooled key 计算 chunk 得分，但额外集成了 KV Cache Management（SnapKV 风格的重要性逐出）以实现 constant decoding memory
- MoBA 在 autoregressive decoding 中 KV cache 随序列长度线性增长；RWKV-X 通过 cache 压缩保证 O(1) 解码内存

涉及论文标题：
- RWKV-X__A_Linear_Complexity_Hybrid_Language_Model

---

## Block-Sparse Attention Pattern (块稀疏注意力模式)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Block-Sparse（BS）注意力模式是 MInference 论文识别的三种长上下文注意力稀疏模式中**最动态**的一种。其注意力权重在空间上没有明显的垂直条纹或斜线规律，而是呈现出分散的块状聚集（spatial clustering）——重要 token 以群体形式出现，但群体的位置高度依赖输入内容。

BS 模式的关键特征：(1) **空间分布**：Dynamic structured——注意力权重分散但存在块级空间聚集（block-level spatial clustering），相邻 token 的重要性往往相近；(2) **GPU 延迟**：Low——64×64 block-level 的 top-K 选择可以直接使用 Block-Sparse FlashAttention kernel，延迟与 block 数量线性相关；(3) **索引构建时间**：Small——使用 mean pooling 降采样后的 block-level matmul 进行近似，开销约占总时间的 25%（高于 VS 的 5-15%，因为需要额外的 pooling + block-level matmul）。

BS 模式的 motivation：MInference 分析发现，即使在注意力最分散的 head 中，非零注意力权重与其最近邻非零权重的距离仍然集中在 ~5 个 token 以内（在 128K context 下），证明了块级空间聚集的存在。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Block-Sparse Head 的动态稀疏索引构建与计算（Algorithm 3）
输入: Q, K, V ∈ R^{S×d_h}, k_b=100, block_size=64

# Step 1: Mean pooling 降采样（block_size=64）
Q̂ = MeanPool(Q, block_size)    # [S/64=2048, d_h] — 沿 seq 维度每 64 行取平均
K̂ = MeanPool(K, block_size)    # [2048, d_h]

# Step 2: Block-level 注意力近似
# 关键性质: MeanPool(Q)·MeanPool(K)^T ≈ MeanPool(QK^T)
# 即 pooling+matmul 近似等价于 matmul+pooling
Â = softmax(Q̂ @ K̂^T / √d + m_causal)  # [2048, 2048]

# Step 3: 提取 top-k_b blocks（每行取 top-k）
i_b = argtopk(Â, k_b, dim=1)           # [2048, 100] — 每行 top-100 blocks

# Step 4: 构建稀疏格式
i_b = sparseformat(i_b)                # 每行 block index list

# Step 5: 稀疏注意力计算
A = softmax(sparse(Q @ K^T, i_b) / √d)
y = sparse(A @ V, i_b)
```

**具体例子**（LLaMA-3-8B, 128K context, BS head, k_b=100, block_size=64）：
- 估计阶段：Q̂ @ K̂^T → [2048, 2048]，仅 $2048^2 \times 128$ FLOPs vs $131072^2 \times 128$
- 稀疏计算：每行仅计算 top-100 blocks（100 × 64 × 64 tokens），共 $2048 \times 100 \times 64^2 \times 128$ FLOPs
- 总 FLOPs: ~$5.4 \times 10^9$（vs dense 的 $2.2 \times 10^{11}$）
- 稀疏率: ~97.5%
- 1M context 下 kernel 级加速：30× vs FlashAttention（三种模式中最快）

术语一般如何实现？如何使用？

实现基于 Triton 版 FlashAttention kernel 修改：以 selected block index 为额外输入，每个 thread block 不再遍历所有 K/V blocks，而是仅遍历每行的 top-K blocks。速度比公式为 $s_p = S / (2B \times k_b)$，其中 B=64 为 block size，$k_b$ 为每行保留的 top blocks。

BS 模式主要分布在模型的 intermediate-to-late layers。适合处理高度动态的注意力需求（如 KV retrieval、multi-hop tracing），但单独使用效果不佳（仅 BS 模式在 InfiniteBench 上平均 18.7 vs Full 38.2），需要与 A-shape 和 VS 模式组合使用。其优势是 kernel 速度最快（30×），且 block-level 的结构化稀疏在 GPU 上非常高效。

**Sparse Frontier 论文的补充发现**：Block-Sparse 使用更小的 block size (16×16，而非 MInference 的 64×64)，因消融实验显示更小 block 始终产生更好性能。Block-Sparse 在 High Scope 或 High Dispersion 任务（如 Ruler VT、Story Filtering）上优于 Vertical-Slash——因为为每个 query block 选择不同的 key token blocks，支持处理多个独立语义片段。Paper 中 block-sparse 始终保留 attention sinks（第一个 key block）和局部上下文（对角线 block）。

涉及论文标题：
- MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention
- The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs
- XAttention: Block Sparse Attention with Antidiagonal Scoring

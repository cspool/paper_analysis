## Dynamic Sparse Attention (动态稀疏注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

动态稀疏注意力（Dynamic Sparse Attention）是一种推理时（inference-time）技术，根据当前输入动态决定注意力矩阵中哪些位置需要计算、哪些可以跳过，从而减少注意力计算的 FLOPs。与静态稀疏注意力（如 Longformer 的固定 sliding window + global attention pattern）不同，动态稀疏注意力不预设固定的稀疏掩码位置，而是根据每个具体 prompt 的内容在线估计重要的 token/block 位置。

MInference 论文将动态稀疏注意力形式化为：$A(M) = \text{Softmax}(QK^T/\sqrt{d} - c(1-M))$，其中 $M_{i,j} \in \{0,1\}$ 是动态稀疏掩码，$c$ 是大常数（如 1e5），使 $M_{i,j}=0$ 的位置的注意力权重接近零。优化目标是最小化稀疏输出与 dense 输出的差异（$\min |A(M) - A_{\text{dense}}|$）以及总延迟（$\min t_{\text{sparse}}(M) + t_{\text{overhead}}(M)$）。

核心挑战在于：(1) 注意力分布高度动态——同一 token 位置在不同 prompt 下关注的 token 完全不同（MInference 验证：对 128K context 取 top-4K 列，在另一 prompt 上 recall 从 96.8% 降至 83.7%）；(2) 但注意力模式的类型（pattern type）在同一 head 上跨 prompt 保持一致性——即 head 总是表现为 A-shape/Vertical-Slash/Block-Sparse 中的某一种；(3) 在线估计必须低开销，否则估计开销抵消稀疏计算的收益。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

动态稀疏注意力在 MInference 的三步 pipeline 中执行：

```
# Step 1: 离线 Pattern Assignment（一次性）
for each attention head h:
    p_best[h] = KernelAwareSearch(Q_ref, K_ref, V_ref, target_FLOPs)
    # 为每个 head 分配 A-shape / Vertical-Slash / Block-Sparse 模式

# Step 2: 在线动态稀疏索引构建（每个 prompt）
for each attention head h:
    if p_best[h] == "A-shape":
        # 静态掩码：固定保留 1K global + 4K local，零开销
        M[h] = StaticMask(global=1024, local=4096)
    elif p_best[h] == "Vertical-Slash":
        Â = softmax(Q_{[-64:]} @ K^T / √d)    # 仅用最后 64 个 query
        i_v = argtopk(sum_v(Â), k_v)           # top-k 垂直列
        i_s = argtopk(sum_s(Â), k_s)           # top-k 斜线
        M[h] = SparseFormat(i_v, i_s)
    elif p_best[h] == "Block-Sparse":
        Q̂ = MeanPool(Q, 64); K̂ = MeanPool(K, 64)
        Â = softmax(Q̂ @ K̂^T / √d)             # block-level 近似
        i_b = argtopk(Â, k_b)                  # top-k blocks
        M[h] = SparseFormat(i_b)

# Step 3: 稀疏注意力计算
for each attention head h:
    y[h] = SparseAttention(Q, K, V, M[h])  # 仅计算 M[h] 标记的位置
```

**具体例子**（LLaMA-3-8B, 128K context）：
- Full attention: $QK^T$ 矩阵 $131072 \times 131072$，约 $2.2 \times 10^{11}$ FLOPs
- Dynamic Sparse Attention: 仅计算 ~4% 的 attention 位置（~96% sparsity），FLOPs 降为 $\sim 9 \times 10^9$
- 开销：Vertical-Slash head 的估计开销 <15%，Block-Sparse head 的估计开销 <25%

术语一般如何实现？如何使用？

动态稀疏注意力的实现通常包含三个关键组件：

1. **模式识别器（Pattern Identifier）**：离线分析 attention head 的稀疏模式类型。可以是基于启发式的（如 MInference 的 kernel-aware search）或基于统计的（如观察 attention map 的空间分布特征：初始 token 集中度、垂直条纹、块状聚集等）。

2. **在线估计器（Online Estimator）**：用极低的计算代价预测当前输入的稀疏分布。常见方法：
   - MInference VS head: 仅使用最后 $\text{last}_q$ 个 query（默认 64）做 matmul
   - MInference BS head: mean pooling + block-level matmul
   - Quest: 基于 query-aware 的 chunk-based importance scoring
   - SparQ Attention: 使用 low-rank hidden states 近似注意力

3. **稀疏计算 kernel**：执行带动态稀疏掩码的高效注意力计算。需要在 GPU 上支持非规则内存访问模式，通常基于 FlashAttention 的 tiling 框架修改。

使用场景：长上下文 LLM 推理的 pre-filling 阶段（prompt >32K tokens 时收益显著），尤其适用于 retrieval、summarization、long-document QA 等需要全局上下文的场景。在 short context（<10K）下动态索引构建开销占比可能达到 30%，收益有限。

涉及论文标题：
- MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention

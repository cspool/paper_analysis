## Dynamic Sparse Index Building (动态稀疏索引构建)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

动态稀疏索引构建（Dynamic Sparse Index Building）是 MInference 推理 pipeline 的第二步，在模型推理时根据当前输入动态构建稀疏注意力掩码的索引结构。其目标是以最小的计算开销（$t_{\text{overhead}}$）估计出尽可能准确的稀疏分布，使得后续的稀疏注意力计算既快又准。

与静态稀疏（如 StreamingLLM 的固定掩码）不同，动态索引构建需要在每个 prompt 的 pre-filling 阶段实时执行。MInference 针对不同的 attention head 模式设计了两种低开销的在线估计方法：Vertical-Slash 的 query-tail 估计和 Block-Sparse 的 mean pooling 估计。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。

**Vertical-Slash Index Building Kernel**（Algorithm 4，简化）：

```
# GPU 并行：每行 block 独立计算索引
输入: i_v ∈ N^{k_v}, i_s ∈ N^{k_s}, block_size B=64

Parallel for row_idx i ← 1 to N (N = ceil(S/B)):
    # 排序索引
    Sort i_v ascending; Sort i_s descending

    # 找到第一条穿过当前行 i 的斜线
    j_s = biset_left(i_s, i × B)       # 二分查找

    # 计算斜线在当前行的范围
    r_start = (i-1) × B - i_s[j_s]
    r_end = i × B - i_s[j_s]

    # Point-range two-way merge
    j_v = 1
    blocks_i = []; columns_i = []
    while j_s ≤ k_s:
        if j_v ≤ k_v and i_v[j_v] < r_end:
            if i_v[j_v] < r_start:            # 垂直点在斜线范围外
                columns_i.append(i_v[j_v])    # → 记录为 column index
            j_v += 1
        else:
            j_s += 1
            if (i-1)×B - i_s[j_s] > r_end:    # 斜线不连续
                # 记录上一段斜线为 block index
                for s from r_start to r_end step B:
                    blocks_i.append(s)
                # 更新范围
                r_start = (i-1)×B - i_s[j_s]
                r_end = i × B - i_s[j_s]
            else:                              # 斜线连续
                r_end += B                     # 扩展范围

    输出: blocks_i (block indices), columns_i (column indices)

# 时间复杂度: O(k_v + k_s) per row
# GPU 并行: 2048 行 (128K/B=64) 同时执行
```

**Block-Sparse Index Building**：
```
# CPU/GPU: Mean pooling + block-level matmul
Q̂ = MeanPool(Q, 64)                         # [2048, d_h]   → 开销: S/B × d_h
K̂ = MeanPool(K, 64)                         # [2048, d_h]
Â = softmax(Q̂ @ K̂^T / √d + m_causal)       # [2048, 2048]  → 开销: (S/B)² × d_h

# GPU 并行：每行取 top-k blocks
Parallel for row i ← 1 to 2048:
    i_b[i] = argtopk(Â[i], k_b=100)

# 索引构建后，转换为 sparse format 供 kernel 使用
```

术语一般如何实现？如何使用？

实现注意事项：
1. **开销权衡**：索引构建开销（VS: 5-15%, BS: ~25%）随 context 长度增长占比下降——因为稀疏计算节省的时间随 O(S²) 增长，而索引构建开销随 O(S)（VS）或 O((S/B)²)（BS）增长
2. **精度 vs 速度**：last_q 越大估计越准确但开销越大（MInference 默认 64，在精度和速度间取得平衡）
3. **Memory 管理**：索引需要存储在 GPU memory 中，1M context 下 <160MB

使用场景：仅适用于长上下文场景（>32K tokens）。对于短 context (<10K)，索引构建开销占比过高（可达 30%），可能抵消稀疏计算收益。

涉及论文标题：
- MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention

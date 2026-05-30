## Re-Index Vector (for Expert-Specific CUDA Kernels)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Re-Index Vector 是 HEXA-MoE 中为 Expert-Specific Operators（ESMM、ESS、ESTMM）CUDA kernel 提供 I/O 指导的辅助数据结构。由于 MoE 的 token 到 expert 分配是动态和不规则的，直接做 expert-wise 计算无法利用 GPU 的合并内存访问（coalesced memory access）和 Tensor Core。Re-Index Vector 通过将 routing choice 信息编码为排序后的 token index 序列，使同 expert 的 token 在逻辑上连续排列，从而在 CUDA kernel 中实现规则的内存访问模式。

构造过程：(1) 统计每个 expert 的 token 数量 ctr[e]（atomicAdd）；(2) 将 ctr[e] 向上取整到 tiling size BLK 的倍数；(3) 计算累积偏移 idx[e]（prefix sum，idx[0]=0，idx[e]=Σ_{j<e} ctr[j]）；(4) 按 routing choice 将原始 token index 写入 v[idx[R[i]]++] = i；(5) 每 expert 的 BLK 对齐尾部填充 -1（表示跳过）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Re-Index Vector 构造 (Algorithm 1, CUDA pseudocode)
Input: R [N] (routing choice, 0 ≤ R[i] < E)
Output: v [N'] (re-index vector, N' ≥ N, divisible by BLK, -1 padding)
        idx [1+E] (start index per expert, idx[E] = N')

// Step 1: 统计 per-expert token 数
ctr[0..E-1] = {0}
parallel for i = 0 to N-1:
    atomicAdd(ctr[R[i]], 1)

// Step 2: 对齐到 BLK
parallel for i = 0 to E-1:
    ctr[i] = BLK * ceil(ctr[i] / BLK)
N' = sum(ctr[0..E-1])

// Step 3: prefix sum → idx
idx[0..E-1] = prefix_sum(ctr[0..E-1])  // idx[1..E] 为各 expert 起始位置

// Step 4: 写入 token indices
parallel for i = 0 to N-1:
    pos = atomicAdd(idx[R[i]], 1)
    v[pos] = i
// v 中尾部填充位置保持 -1

// ESMM Kernel 中使用 Re-Index Vector:
parallel for i in range(0, N', BLK):    // i 步进 BLK
    exp = R[v[i]]                         // 当前 BLK 对应的 expert
    parallel for j in range(0, D2, BLK):  // 输出维度 tiling
        c = b[exp, j:j+BLK].repeat(BLK, 1)  // 加载 bias
        for k in range(0, D1, BLK):       // 输入维度 tiling
            parallel for t = 0 to BLK-1:
                xsub[t] = (v[i+t] != -1) ? x[v[i+t], k:k+BLK] : 0
            wsub = w[exp, k:k+BLK, j:j+BLK]
            c += xsub @ wsub              // Tensor Core MMA (16×16×16)
        parallel for t = 0 to BLK-1:
            if v[i+t] != -1:
                y[v[i+t], j:j+BLK] = c[t]  // 原位写回
```

Re-Index Vector 的关键作用：(1) 将同 expert 的 token 聚集到连续区域，使 thread-block 只需加载一次 expert 权重（而非每个 token 加载一次）；(2) 通过 padding -1 使所有 expert 的 token 数对齐到 BLK，保证规则的内存访问 pattern；(3) 输出按 v 中的原始 index 写回，保持与输入相同的 token 顺序。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Re-Index Vector 在每个 MoE 层的前向和反向传播中动态构建，开销为 O(N) 的 atomic 操作。由于 BLK 对齐引入了少量 padding（最多 BLK-1 per expert），padding 位置在 kernel 中通过检查 v[i+t] != -1 跳过计算和写回，不产生冗余 FLOPs。ESTMM 中两输入共享同一 re-index vector（因为它们来自同一个 ESMM 输出），进一步减少构造开销。开源实现：https://github.com/UNITES-Lab/HEXA-MoE。

涉及论文标题：
- HEXA-MoE: Efficient and Heterogeneous-aware MoE Acceleration with ZERO Computation Redundancy

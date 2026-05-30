## Top-p via Binary Search (GPU Kernel)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Top-p via Binary Search 是 Twilight 在 GPU 上高效实现 top-p attention weight 选择的并行算法。直接的 top-p 实现需要降序排序后累积——在 GPU 上极为低效（O(N log N) sorting）。该算法采用 parallel-friendly binary search：在 [0, max(W)] 区间二分搜索阈值，每次迭代通过 tensorized element-wise 操作计算累积概率并比较，避免排序且不物化中间变量。收敛到精度 ε 需 O(log(range/ε)) 次迭代（typically 8-12 次）。

从kernel调度角度拆解术语，给出具体例子。
```
// Algorithm 1: Top-p via Binary Search (GPU kernel, fully tensorized)
Input: W ∈ R^{BS×H×N} (normalized attention weights), threshold p, tolerance ε
Output: I (selected indices), M ∈ {0,1}^{BS×H×N} (mask)

l = 0, r = max(W)  // 所有element-wise操作融合为单次循环

repeat:
  m = (l + r) / 2
  
  // Fused operations (tensorized on GPU, single pass):
  // ① where(W < m, 0, W) — mask below threshold
  // ② sum(masked_W)      — compute cumulative probability
  masked = where(W >= m, W, 0.0)
  cumsum = sum(masked)
  
  if cumsum >= p: l = m  // cumulative enough → raise threshold, prune more
  else: r = m            // not enough → lower threshold, keep more
  
until max(W[W > r]) - min(W[W >= l]) < ε

M = (W >= l)  // final mask
I = indices(M == 1)
return I, M
```

关键优化：(a) element-wise max/where/sum 融合为单次 register-level 循环，不物化中间变量（如 W0）；(b) 8-12 次迭代即可收敛（ε=0.01）；(c) 比 sorting-based 方法快 O(N) 倍。

术语一般如何实现？如何使用？
修改 FlashInfer 的 top-p sampling kernel（原用于 LLM text generation token sampling），应用场合从 "选下一个 token" 变为 "选 attention weights"。使用 CUDA thread block 并行化，每 thread 处理一部分 token 的 element-wise 操作。适用于所有需要 top-p selection 的 GPU kernel。

涉及论文标题：
- Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning

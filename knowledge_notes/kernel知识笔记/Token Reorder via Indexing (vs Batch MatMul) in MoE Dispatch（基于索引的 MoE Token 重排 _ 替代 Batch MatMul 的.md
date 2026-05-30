## Token Reorder via Indexing (vs Batch MatMul) in MoE Dispatch（基于索引的 MoE Token 重排 / 替代 Batch MatMul 的 Dispatch 实现）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Token Reorder via Indexing 是 Huang et al. (NeurIPS 2024) 提出的用高级索引（advanced indexing）替代 batch matrix multiplication (bmm) 来重排 MoE tokens 的 kernel 优化。在 Static Gating 中，token dispatch 和 reorder 通过构建稀疏 dispatch mask 矩阵并与 token tensor 进行 batch matmul 实现——O(S²EDC) 复杂度，大部分计算浪费在 ×0 操作上。论文提出用 torch.argsort + torch.index_select (或 Python advanced indexing) 直接重排 tokens——O(SD + S log S) 复杂度，纯 memory-bandwidth bound 操作。

从kernel调度角度拆解术语：

```
// === Static Gating: Batch MatMul Dispatch ===
// Input: tokens X ∈ R^{S×D}, mask M ∈ R^{E×S×S×C}
// M 极度稀疏: 仅 S×k 个 1s (k=top-k), 其余为 0
dispatched = torch.bmm(M, X)  // cuBLAS SGEMM kernel
// → GPU: SGEMM tiles, loads both X and M into shared memory
// → M 中大量零值被加载并参与乘加 → 浪费 FMA + bandwidth
// → S=8, E=512, D=1024: ~860M FLOPs, 92%为×0

// === Dynamic Gating: Indexing Dispatch ===
sorted_idx = torch.argsort(expert_ids)          // RadixSort kernel: O(S log S)
sorted_X = X[sorted_idx]                        // Gather kernel: O(SD) BW
sizes = torch.bincount(expert_ids, minlength=E) // Reduce kernel: O(S)
batches = torch.split(sorted_X, sizes)          // View/slice: O(1)

// → 无 matrix multiplication
// → Gather kernel: 仅读取 X[sorted_idx[i]] 写入 sorted_X[i]
// → Memory BW: 2×S×D×4 bytes (read X + write sorted_X)
// → 比较 Static bmm: S²EDC × 4 bytes (包括 zeros)
```

关键 Kernel 分析：
```
Advanced Indexing (X[sorted_idx]) GPU kernel:
  grid: ceil(S*D / block_size) blocks
  每 thread: load sorted_idx[tid/D] → compute X offset → load X[offset] → store
  
  vs Batch MatMul GPU kernel (cuBLAS):
  grid: 2D tiling over (E*S) × D
  每 block: load M tile + X tile → compute M·X → 大量 zeros
  Waste: tile overhead + FMA for zeros
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

PyTorch 实现：`dispatched = tokens[argsort(assignments[:, 1])]`。底层调用 GPU gather kernel（`index_select` 或 `take_along_dim`），memory-bandwidth bound（vs compute-bound batch matmul）。适用条件：当 S>32 时优势显著；当 S 极小时（<8），gather kernel launch overhead 可能超过 bmm 的固定开销。论文实验表明 batch=80 时 Dynamic Gating 优于 Megablock (BCSR sparse matmul) 1.46×，因 dense matmul + indexing 比 sparse matmul 在 GPU 上更高效。

涉及论文标题：
- Toward Efficient Inference for Mixture of Experts
- Towards MoE Deployment: Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference

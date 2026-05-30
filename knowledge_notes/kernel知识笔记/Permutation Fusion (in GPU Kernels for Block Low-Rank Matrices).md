## Permutation Fusion (in GPU Kernels for Block Low-Rank Matrices)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Permutation Fusion是将tensor维度重排（transpose/permute/reshape）与计算操作融合到单个GPU kernel中的技术。在BLR上下文中解决：Monarch和BLAST的PyTorch基线中，permutation需要独立kernel launch——创建与输入同大小的新tensor并按新顺序写入元素。当在contiguous维度排列时访问变为uncoalesced，DRAM bandwidth利用率骤降。融合的基本原理：不在global memory中物理重排数据，而在tile加载时计算permuted index并将结果直接写入目标layout。

从kernel调度角度拆解术语：
以Monarch ② fused perm+bmm为例（b₂↔b₁ + bmm with V^T = single kernel）：

```
// 需要实现: (b₁, n, r'b₂) → (n, b₂·n·r') 同时完成bmm
// 原本: 1 kernel for bmm → output [b₁, n, r'b₂]
//       + 2 kernels for permutations → 中间128MB张量
// 融合后: 1 kernel → 输出直接为 [n, b₂·n·r']

// Permutation index计算逻辑:
// 目标: 输出中位置 (n_idx, flat_rank_idx) 对应
//   b_2 = flat_rank_idx // n // r'  (or: r_range // r')
//   r'_offset = (r_range % r') + b_1 * r'
//   最终写入: Z'[n_idx, b_2 * n * r' + r'_offset]

// Triton实现: 通过program_id和range计算输出块位置
// → 计算需要加载的输入位置（inverse permutation）
// → 加载数据 → dot() → 直接写入目标位置
// 零额外kernel launch, 零额外global memory allocation
```

BLAST ⑤ transpose-based permutation消除：
```
// 转置S/U: S^T[r, b₁, b₂], U^T[r, b₂, q] (offline)
// 从左乘: [b₁, n, r] · S^T · U^T
// 每个kernel内部transpose中间输出tile
// 全部使用triton.dot()保持tensor core → 零独立permutation kernel
```

术语一般如何实现？如何使用？
Permutation fusion要求permutation是静态已知的（非data-dependent）。在Triton中通过index arithmetic实现：输出tile写入前计算permuted addresses→直接写入。对Monarch/BLAST，所有permutation都是固定维度重排，满足条件。不适用：(a) data-dependent scatter/gather；(b) 需全局reduction的操作（如softmax）。性能提升幅度取决于原permutation的数据移动量——序列越长（n越大），消除的permutation数据移动越多，speedup越显著。

涉及论文标题：
- Memory-Efficient Acceleration of Block Low-Rank Foundation Models on Resource Constrained GPUs

---

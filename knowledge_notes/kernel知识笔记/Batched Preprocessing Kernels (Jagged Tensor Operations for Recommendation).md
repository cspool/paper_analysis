## Batched Preprocessing Kernels (Jagged Tensor Operations for Recommendation)

术语是什么？
Batched Preprocessing Kernels是处理推荐系统data preprocessing pipeline中irregular/jagged tensor操作的GPU/Accelerator kernel。与regular dense tensor GEMM不同，这些kernels处理：(1) jagged tensors (variable-length per sample)；(2) data-dependent control flow (binary search, conditional matching)；(3) nested indexing (batch→user→event→feature)。KernelEvolve通过自动生成fused Triton kernel解决emerging accelerators上native operator coverage不足的问题。

从kernel调度角度拆解术语：
**MapIdTransform** (Fused on MTIA):
```
PyTorch: bucketize → clamp → gather → where (4独立ops, MTIA v2i上部分op缺失需CPU fallback)

KernelEvolve Triton (single launch, MTIA-optimized):
  for block in parallel:
    values_tile = tl.load(coalesced, mask)
    // Compile-time loop unrolling: for _ in range(20) — supports up to 2^20 mappings
    left=0, right=|M|
    for _ in range(20):
        mid = (left+right)>>1
        left = tl.where(search_active & (values_tile > M[mid]), mid+1, left)
        right = tl.where(search_active & (values_tile <= M[mid]), mid, right)
    output = tl.where(M[left] == values_tile, left+1, 0)  // in-register match
    tl.store(coalesced, output, mask)
  
MTIA v2i: 3.28-4.07× speedup; v3: 1.05-1.36× (stronger baseline)
```

**MBDT** (SIMD-vectorized on MTIA):
```
PyTorch: per-feature, per-element torch.bucketize

KernelEvolve: SIMD-vectorized counting replaces binary search
  for border_val in borders:
      count += (values > border_val).to(int)  // 64-256 elements simultaneously
  // For 3-10 element border arrays: O(n) > O(log n) due to branch-free + no CF overhead
  
MTIA v2i: 2.94-9.25×; v3: 2.31-3.09×
```

**Batch Event Truncate** (Multi-feature parallel):
```
PyTorch: per-feature sequential loop (no batched variant existed)

KernelEvolve batched Triton: all features in parallel, single launch
  No truncation needed: 9.8-14.5× (single vectorized compare vs per-element loop)
  Truncation required: 1.4-2.0× (constant launch vs sequential iteration)
```

术语一般如何实现？如何使用？
KernelEvolve通过graph-based search自动生成fused preprocessing kernels。对MTIA v2i（native operator coverage不足），生成的kernels不仅是性能优化，更是functional enablement——唯一可行的on-device执行路径。对v3（coverage更完整），kernel fusion和hardware-specific tuning仍提供2-3× speedup。关键MTIA-specific优化：compile-time loop unrolling (for branchless binary search)、coalesced block-parallel execution、register-resident computation (no intermediate tensor allocation)、avoiding tl.where in loops (direct boolean-to-int conversion)。

涉及论文标题：
- KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta

---

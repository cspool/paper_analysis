## Decomposition-based Computation-Communication Overlapping

术语是什么？

Decomposition-based overlapping（基于分解的重叠）将 GEMM 输出 tensor 沿单一维度分解为多个子 tensor，依次执行 subtensor_i 的 GEMM → subtensor_i 的通信 → subtensor_{i+1} 的 GEMM，实现计算和通信的异步交叠。代表性工作：CoCoNet (ASPLOS'22)、Centauri (ASPLOS'24)、Async-TP (PyTorch)、Domino、MegaScale。

从系统架构角度拆解术语：

关键限制——单维分解 vs tile 的非连续性：

```
GEMM output C[M×N], 沿 M 维分解: subtensor_i = C[i*chunk_M : (i+1)*chunk_M, :]
  → 地址连续 ✓ (row-major) → 可调用 NCCL
  → 但 tile 在地址上天然二维非连续（stride=N）
  → 单维 subtensor 无法对齐 tile 粒度 → 无 tile-wise overlapping ✗
  → chunk_M 较小时 GPU SM 利用率下降 ✗

根本矛盾: NCCL 要求连续地址 → 只能单维分解
        GEMM tile 是二维非连续的 → decomposition 粒度 > tile 粒度
        → 永远无法实现 tile-wise overlapping
```

术语一般如何实现？如何使用？

PyTorch Async-TP 通过 `_register_async_allreduce()` 注册异步通信。CoCoNet 用 compiler-based 方法自动生成协调 kernel。Centauri 构建 communication partition space 做 hierarchical scheduling。优势是实现简单——直接调用 cuBLAS 和 NCCL API，无需修改 kernel。劣势是重叠粒度粗、碎片化 GEMM 可能降低 GPU 利用率。

涉及论文标题：
- Efficient and Adaptable Overlapping for Computation and Communication via Signaling and Reordering

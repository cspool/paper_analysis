## Wave Specialization (Producer-Consumer) on AMD

术语是什么？
Wave specialization（producer-consumer scheduling）是 GPU kernel 调度模式：将 wave/warp 分为专职 producer（仅做 HBM→shared memory 数据搬运）和 consumer（仅做 shared memory→compute）。在 NVIDIA GPU 上占主导（FlashAttention-3、CUTLASS、TK），通过 TMA、wgmma、mbarrier 等硬件特性实现高效 overlap。但在 AMD CDNA GPU 上，wave specialization 因架构差异而性能退化。

从kernel调度角度拆解术语：
在 AMD 上 wave specialization 的限制：
1. 静态寄存器分配：每 SIMD 512 寄存器在所有 wave 之间平均分配，producer wave 只需少量地址计算寄存器但仍占用大量 register，consumer wave 无法回收，缩小了 output tile size 和 arithmetic intensity。
2. 无 TMA/wgmma/mbarrier：缺少异步 HBM→shared memory 搬运（buffer_load 仍需 s_waitcnt 同步）、无 shared memory 直接矩阵乘、software barrier 开销大。
论文 Table 2 验证：随 producer 数量增加，性能下降（4P+8C: 893 TFLOPS，0P+8C: 1281 TFLOPS）。因此在 AMD 上推荐 8-WAVE PING-PONG（无专职 producer）或 4-WAVE INTERLEAVE。

术语一般如何实现？如何使用？
在 NVIDIA 上通过 TMA（cp_async_bulk）+ wgmma + mbarrier 实现。在 AMD 上，论文证明不使用 wave specialization（0 producer）配合 8-WAVE PING-PONG 性能最优。HK 通过模板参数控制调度模式。

涉及论文标题：
- HipKittens: Fast and Furious AMD Kernels

---

## Kernel Fusion for Compute-Intensive GEMM Chains (计算密集算子链融合)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Kernel Fusion for Compute-Intensive GEMM Chains 是将 LLM/CNN 中连续多个 GEMM 操作（如 FFN 层的 GEMM1→GEMM2 或 Gated FFN 的双分支 GEMM→GEMM2）合并为单个 CUDA kernel 的编译优化技术。与传统的 compute-activation 融合（将 GEMM 与 memory-intensive 的激活函数/逐元素操作融合）不同，compute-intensive operator chain fusion 的目标是在片上保留完整的中间张量（通常远大于单 SM 的 shared memory 256KB），避免 HBM round-trip。现有方法受限于单 SM 的 SMEM 容量（如 Chimera 最多 227KB/SM on H100），当中间张量超出此限制时 fusion 失败。FlashFuser 首次利用 H100 的 DSM（cluster 内多 SM SMEM 互联，~3.6MB）扩展 fusion 边界。

从编译框架角度拆解术语：
FlashFuser 的 fusion 编译流程：
```
DNN Graph (GEMM chain)
  ↓
[Fusion Search Engine] — 枚举 LoopSchedule × TilingSize × ResourceMapping
  ↓ pruning (5 rules: Divisible Tile, Cluster Size≤16, Activation,
  |  Dependency≠spatial L, Memory Capacity)
  ↓
[Dataflow Analyzer] — 量化跨 reg→SMEM→DSM→L2 的 data movement
  ↓ Top-K=11 candidates
[Cost Model] — minmax bottleneck: min max_l (V_l / B_l)
  ↓
[Hardware Profiling] — 在 H100 实测 Top-K 选最优
  ↓
[CUTLASS Code Generation] — prologue + mainloop (dsm_comm注入) + epilogue
  ↓
Fused CUDA Kernel — single kernel 执行 GEMM0→GEMM1→Store
```

传统方法对比：
- cuBLAS/PyTorch: 完全无 fusion，2 次独立 GEMM kernel，中间 tensor 经 HBM round-trip
- TASO/TVM Relay: graph substitution 但不支持 sequential GEMM 的 compute-intensive fusion
- Chimera/BOLT: SMEM-based fusion，SMEM 容量限制导致大 GEMM chain 可能 fusion 失败
- FlashFuser: DSM-based fusion，利用 cluster 内多 SM SMEM 互联扩展 on-chip memory

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FlashFuser 通过：(1) dsm_comm primitive 统一描述 cluster 内 inter-SM data exchange pattern；(2) Dataflow Analyzer 贪心 spill 策略将 reusable tensor 从 reg→SMEM→DSM 逐级放置；(3) Search Engine 在 DSM 显著扩大搜索空间（~10^4 → ~10^6 for GPT-6.7B）后通过 DSM-aware pruning rules 和 cost model 高效搜索。运行时通过 M-dimension binning + table lookup 选择预编译 kernel。在 GEMM chains 上 3.1× over PyTorch, 4.1× over SOTA compilers；全局显存访问减少 58%；端到端 SGLang serving 1.24× speedup。

涉及论文标题：
- FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection

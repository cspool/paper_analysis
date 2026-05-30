## Fused Kernel for Communication-Computation Overlap in MoE Inference

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fused Kernel 是 MegaScale-Infer 中两类 kernel fusion 优化：(1) TP Communication-Computation Fusion：使用 Flux（ByteDance kernel fusion 库）将 tensor parallelism 的 all-gather/reduce-scatter 与相邻 GEMM 融合为单 CUDA kernel，利用 NVLink P2P load 实现 zero-copy；(2) Sequential Memory-Intensive Operator Fusion：将 gating + top-k selection + per-expert count + token scatter 等多个 memory-bound 操作融合为单 kernel，消除多次 kernel launch 和中间 global memory access。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
TP All-Gather + GEMM Fusion（Flux 风格）：
```
__global__ void fused_allgather_gemm(input_chunk, weight, output) {
    // Phase 1: NVLink P2P load 从 peer GPU 读取远程 chunk 到 shared memory
    // Phase 2: 直接在 shared memory 上执行 GEMM Tensor Core MMA
    // 无需等待 All-Gather 完成——通信与计算在寄存器/shared mem 级别融合
}
```

Sequential Operator Fusion（Gating + Top-K + Scatter）：
```
__global__ void fused_gating_pipeline(h, W_gate, tokens, expert_inputs) {
    // 1. Per-token gating: scores = dot(h[tid], W_gate)  // registers
    // 2. Top-K partial sort in registers
    // 3. Atomic scatter to expert buffer (shared mem + global atomics)
    // 4. Shared memory count reduce to global
    // 4+ kernel launches → 1 kernel launch, 无中间 global memory roundtrip
}
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- Flux：ByteDance 开源的 CUDA kernel fusion 库，在 kernel 内使用 NVLink P2P ld_volatile 从 peer GPU 加载数据。
- 适用：intra-node TP（NVLink 高带宽低延迟），跨节点不适合（InfiniBand 延迟高、无 P2P load）。
- 效果：未单独评估，综合在 MegaScale-Infer 整体性能中。

涉及论文标题：
- MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism

---

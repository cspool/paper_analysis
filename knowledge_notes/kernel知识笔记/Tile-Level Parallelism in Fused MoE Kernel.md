## Tile-Level Parallelism in Fused MoE Kernel

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Tile-Level Parallelism 是 FlashMoE 将大矩阵分解为细粒度、可独立调度的 tile 计算单元的策略。每个 128×64 tile 对应一个独立 task descriptor。MoE FFN (2×GEMM + activation) 和 Combine 统一为 task 抽象：t = (M, ⋆, φ)，执行 F_t(A, B, C, D) := C ← φ(A ⋆_t B + D)。⋆ 为 · (GEMM) 或 ⊙ (Hadamard)。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Tile: 128×64, 128 threads/block
// Task 抽象:
// FFN:  t₁ = (M, ·, SiLU)     → C₁ ← SiLU(A·W₁ + b₁)
//       t₂ = (M, ·, identity)  → C₂ ← A·W₂ + b₂
// Combine: t₃ = (M, ⊙, identity) → C ← A⊙S + C

// Task struct (128-byte cache line aligned):
struct Task {
    const byte* aData;
    array<const byte*, 2> bData;  // W1, W2
    array<byte*, 2> cData;        // output
    uint M, tileIdx, batchIdx, peerIdx, expertIdx;
    TaskType taskType;  // GEMM0 | GEMM1 | Combine
};

// Processor GEMM0: CUTLASS gemm + SiLU epilogue + stage to shared memory
// Processor GEMM1: CUTLASS gemm + NVSHMEM put (if remote)
// Processor Combine: Hadamard product + accumulation
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 128×64 tile = 8192 elements; FP32 tile = 32 KB
- CUTLASS in-kernel device-side GEMM 执行 tile 级矩阵乘
- 同一 expert 的多个 tiles 可由不同 Processor block 并行处理
- Tile dimension selection balance: register usage + shared memory + SM occupancy

涉及论文标题：
- FlashMoE: Fast Distributed MoE in a Single Kernel

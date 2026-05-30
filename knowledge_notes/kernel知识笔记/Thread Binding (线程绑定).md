## Thread Binding (线程绑定)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Thread Binding 是 GPU kernel 编程中将 tile 级操作和数据映射到具体硬件线程（thread）的过程。在 TileLang 中，Thread Binding 是四个 schedule space 之一，核心挑战是：(1) 确定 block 级 register files 如何划分到各 thread；(2) 推断各 buffer 的 Fragment Layout；(3) 确定循环如何正确 parallelize 以匹配 layout 约束。TileLang 通过 Layout Inference Pass 自动处理 Thread Binding：按优先级层次（GEMM > Element-wise > Copy）逐步推断所有 buffer 的 thread mapping，并在无法自动推断时允许用户通过 T.Fragment 手动指定。

从 kernel 调度角度拆解术语，比如术语所在 kernel 调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Thread Binding 的推理过程（以 GEMM + bias add 为例，图 7）：
```
// 场景: C_local[4x4] = GEMM(A_shared, B_shared) + D_bias[4]

// Step 1: GEMM (最高优先级) 确定 thread binding
// C_local 的 Fragment Layout:
//   8 threads, 每 thread 持有 2 elements
//   thread 0: {C[0,0], C[0,2]}
//   thread 1: {C[0,1], C[0,3]}
//   thread 2: {C[1,0], C[1,2]}
//   ...
//   每 row 由 2 threads 处理

// Step 2: Infer D_bias layout from C_local's layout
//   由于每 row 的 2 threads 都需要相同 D[row] 元素:
//   D_bias 需要 replicate: 每个 D[row] 复制到 row 对应的 2 threads
//   Fragment Layout for D: f(row_idx) → (thread_id, reg_id)
//     thread 0,1: D[0] in reg 0 (replicated)
//     thread 2,3: D[1] in reg 0 (replicated)
//     ...

// 生成伪代码:
for tx in T.Parallel(8):    // 8 threads
  for i in T.vectorized(2):  // per-thread vectorized
    C_local[tx//2, (tx%2)*2 + i] += D_bias[tx//2]  // D 已 replicate to each thread
```

图 8 展示了更复杂场景的 multi-stage Thread Binding Inference：T.copy 操作先扩展为多个 loop axes → Layout Inference Pass 自动 parallelize + vectorize → Layout Swizzling 应用 → 最终生成高效 memory access pattern。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 TileLang 中用户通常无需手动处理 Thread Binding — T.Kernel(threads=N) 确定总线程数，Layout Inference 自动处理映射。专家可通过 T.Fragment 和 InferLayout 接口手动定义 thread→buffer 映射策略。T.Parallel 原语可显式标注并行循环维度。

涉及论文标题：
- TileLang: A Composable Tiled Programming Model for AI Systems

---

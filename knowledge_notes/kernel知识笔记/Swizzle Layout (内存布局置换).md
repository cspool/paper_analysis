## Swizzle Layout (内存布局置换)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Swizzle Layout（内存布局置换）是一种 GPU shared memory 地址重映射技术，通过 XOR（异或）位操作将逻辑上连续的 shared memory 地址映射到不同物理 bank，以避免 bank conflict。在 NVIDIA GPU 上，shared memory 被组织为 32 个 bank（每个 bank 4 bytes），同一 warp 内 32 个 thread 如果在同一 cycle 访问同一 bank 的不同地址，会触发 bank conflict（最多 32-way），导致访问被串行化，带宽下降。Swizzle Layout 通过对地址高位和低位做 XOR 操作，打散连续访问的 bank 分布，确保相邻线程访问不同 bank。TileLang 通过 Layout Composition 机制将 SwizzleLayout 作为 built-in strategy，T.gemm 默认对 A_shared 和 B_shared 应用 SwizzleLayout。

从 kernel 调度角度拆解术语，比如术语所在 kernel 调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Swizzle Layout 在 GEMM shared memory tile 中的效果：
```
// 无 Swizzle（4-way bank conflict 示例）:
// 32 threads × 128-bit (4 floats) load → row-major 连续地址
// Thread 0: addr 0   (bank 0)
// Thread 1: addr 4   (bank 1)
// ...
// Thread 8: addr 32  (bank 0) ← conflict with thread 0! (周期 8 × 4 bytes = 32 = bank repeat)

// 有 Swizzle（T.gemm 默认）:
// XOR(bits=3) swizzle: addr' = addr XOR (addr >> 3) & 0x7
// Thread 0: addr 0  → bank 0
// Thread 1: addr 4  → bank 1
// ...
// Thread 8: addr 32 → swizzle → addr' = 32 XOR (32>>3)&7 = 32 XOR 4 = 36 → bank 9  ← no conflict!
// 结果: 0 bank conflict → shared memory bandwidth = peak
```

TileLang 的 Swizzle 实现通过 Layout Composition：
```
base_layout = Layout(shape=[M, K], strides=[K, 1])  // row-major
swizzle_layout = SwizzleLayout(bits=3, dim=1)        // 沿 K 维 swizzle
final_layout = base_layout ∘ swizzle_layout           // composition
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Swizzle Layout 在 CUTLASS 中通过 `cutlass::layout::ColumnMajorSwizzle` 或 `RowMajorSwizzle` 实现。TileLang 中 T.gemm 自动应用，用户可通过 T.annotate_layout 覆盖自定义 swizzle 模式。T.use_swizzle(10) 是一种不同的 swizzle — 它作用于 thread block scheduling ordering（而非 shared memory 地址），通过打乱 thread block 的执行顺序优化 L2 cache locality（相邻 thread block 可能访问重叠的 global memory 数据区域）。

涉及论文标题：
- TileLang: A Composable Tiled Programming Model for AI Systems

---

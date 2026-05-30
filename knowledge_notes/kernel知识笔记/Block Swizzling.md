## Block Swizzling

术语是什么？

Block swizzling 是 CUTLASS GEMM kernel 中使用的一种 tile-to-SM 调度优化技术。它不按 tile 在输出矩阵中的自然行-列索引顺序分配 tile 到 SM，而以 swizzling 方式交错分配以提升 memory access locality 和 L2 cache 效率。例如 swizzling size=2 时，相邻地址的 tile 因 swizzling 而可能在不同 wave 中完成。Block swizzling 是导致 tile 完成顺序与内存地址顺序不匹配的直接原因，也是 FlashOverlap 需要 pre-communication reordering 的根本 motivation。

从kernel调度角度拆解术语：

以 4×4 tile grid、swizzling size=2 为例：

```
原始 tile 索引（row-major）:     实际 SM 分配和执行顺序:
+----+----+----+----+            Wave W1: tile(0,0), tile(0,2) ← 内存地址不连续！
|  0 |  1 |  2 |  3 |            Wave W1: tile(2,0), tile(2,2)
+----+----+----+----+            Wave W2: tile(0,1), tile(0,3)
|  4 |  5 |  6 |  7 |            Wave W2: tile(2,1), tile(2,3)
+----+----+----+----+            Wave W3: tile(1,0), tile(1,2)
|  8 |  9 | 10 | 11 |            Wave W3: tile(3,0), tile(3,2)
+----+----+----+----+            Wave W4: tile(1,1), tile(1,3)
| 12 | 13 | 14 | 15 |            Wave W4: tile(3,1), tile(3,3)
+----+----+----+----+
```

**Annotations**: 128 SM GPU 上每个 wave 包含 128 个 tile（full occupancy）。Swizzling 使 wave 内 tile 地址不连续——decomposition-based 方法因要求连续地址而无法利用 tile-wise overlapping。FlashOverlap 通过 pre-communication reordering 解决此问题。

术语一般如何实现？如何使用？

CUTLASS 中通过 `cutlass::gemm::threadblock_swizzle` 策略配置 swizzling 模式。Block swizzling 通过改善 L2 cache hit rate 提升 memory access 效率——相邻 SM 处理的 tile 在数据空间上也相邻，共享 L2 cache line。CUTLASS profiler 自动选择最优 swizzling size。

涉及论文标题：
- Efficient and Adaptable Overlapping for Computation and Communication via Signaling and Reordering

## Chiplet Swizzling (XCD Swizzle — Cache-Aware Grid Scheduling)

术语是什么？
Chiplet swizzling 是 HipKittens Algorithm 1：在 AMD MI355X 8-XCD chiplet GPU 上，通过 remap thread block 的 grid 坐标，优化两级缓存（每 XCD 私有 4MB L2 + 全局 LLC）的数据复用。默认 row-major grid 下 L2 hit rate 仅 36%-55%；Algorithm 1 通过 XCD grouping（chunks of C blocks 归同一 XCD）和 hierarchical windowed traversal（W 高度的垂直窗口遍历），将 L2 hit rate 提升至 78-79%，整体带宽提升 19%。

从kernel调度角度拆解术语：
```
Algorithm 1 流程:
1. Flatten (b.x, b.y) to linear ID
2. XCD grouping: 连续 C blocks → 同一 XCD
3. Hierarchical windowing: W 行 × num_cols 列的窗口内分配
4. 窗口内优先沿列方向（同一 XCD 覆盖矩形 L2 tile）
5. 跨 XCD 窗口对齐（重叠 A 行和 B 列 → 提升 LLC hit rate）
6. 尾部 block 保持原始顺序
```
参数 W（窗口高度，控制 L2 reuse）和 C（chunk 大小，控制 LLC reuse）的权衡：L2 带宽约 3x LLC 带宽，优先最大化 L2 hit rate。

术语一般如何实现？如何使用？
在 GEMM kernel launch 前 CPU 端执行，将 remap 后的 (row, col) 传给 kernel。当输出 tile 数与 XCD 数互质时收益最大。可推广到其他 workload。

涉及论文标题：
- HipKittens: Fast and Furious AMD Kernels

---

## RowSync Policy

术语是什么？
RowSync 是 cuSync 中平衡同步粒度与开销的策略。它将同一行的所有 producer tile 映射到**同一个** global memory semaphore，semaphore 值表示该行已完成 tile 的数量。Consumer tile 等待其依赖行的 semaphore 达到该行 tile 总数（grid.x）后才开始计算。相比 TileSync（每 tile 一个 semaphore），RowSync 将同步次数从 O(grid.x×grid.y) 降至 O(grid.y)，但 consumer 必须等整行完成而非单个 tile，降低了并发粒度。在大 grid size 的 GeMM 和 Conv2D implicit GEMM 中，RowSync 的全局内存访问减少带来的收益超过并发降低的损失。

从kernel调度角度拆解术语：
```
// RowSync policy: 同行tile共享semaphore
class RowSync {
    int sem(dim2 tile, dim2 grid) {
        return tile.y;  // 仅按行索引，同行所有列tile共享
    }
    int value(dim2 tile, dim2 grid) {
        return grid.x;  // semaphore达到grid.x表示整行完成
    }
};
// MLP示例: producer grid=[H/TileN, B], 2D grid
// Producer tile C(i,j) post: atomicAdd(sems[i], 1)
// Consumer tile E(i,j) wait: while(sems[i] != H/TileN);
// 同步次数: 仅B次(vs TileSync的B×H/TileN次)

// 在B×S=512 GPT-3 MLP中 RowSync减少waves从6到4.8(↓20% time)
```
RowSync 在 B×S≥512 时最优（更多行→更多overlap机会），提升从 4% (B×S=256)到 20% (B×S=512)。对 Conv2D implicit GEMM，RowSync 沿输出 channel 维度同步，效果类似。

术语一般如何实现？如何使用？
提供 `sem()` 返回 `tile.y`，`value()` 返回 `grid.x`。cuSyncGen 自动为每个依赖同时生成 TileSync 和 RowSync，用户实验选择最优策略。RowSync+WRT 变体添加避免 wait-kernel、重排 tile load、避免自定义 tile order 三项优化。cuSync 也支持 StridedSync（用于 Attention 的 strided dependency）和 Conv2DTileSync（Conv2D 专用 per-tile sync）等变体。

涉及论文标题：
- A Framework for Fine-Grained Synchronization of Dependent GPU Kernels

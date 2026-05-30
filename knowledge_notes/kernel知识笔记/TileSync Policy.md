## TileSync Policy

术语是什么？
TileSync 是 cuSync 框架中最细粒度的同步策略（synchronization policy）。它为 producer kernel 的每个 tile 分配一个独立的 global memory semaphore，实现 tile 到 semaphore 的一对一映射。Consumer kernel 的每个 thread block 需要通过 busy-wait 等待其依赖的具体 producer tile(s) 的 semaphore 达到预期值后才能开始计算。在 MLP 场景中，consumer tile E(x,y) 依赖同一行的所有 producer tiles C(x,0), C(x,1), ..., C(x,N-1)，TileSync 要求 consumer 依次等待每个 producer tile 的独立 semaphore。此策略同步次数最多，但提供最大的并发机会，在小 grid size 时表现最优。

从kernel调度角度拆解术语：
```
// TileSync policy: 每个tile一个独立semaphore
class TileSync {
    int sem(dim2 tile, dim2 grid) {
        return tile.x * grid.y + tile.y;  // 行优先线性索引
    }
    int value(dim2 tile, dim2 grid) {
        return 1;  // semaphore=1表示该tile已计算完成
    }
};
// MLP示例: producer grid=[H/TileN, B], consumer grid=[H/TileN, B]
// Consumer E(i,j)依次等待所有同行producer tile:
//   for k in 0..(H/TileN)-1:
//       while(sems[i * (H/TileN) + k] != 1);  // busy-wait
//   __syncthreads();
//   // 所有依赖tile就绪，开始加载和计算
```
TileSync 在 B×S=1~256 的 GPT-3 MLP 时表现最好——此时 grid x 维度只有 1 个 thread block，同步次数少。在 B×S=256 时，TileSync+WRT 比 StreamSync 减少 1 个 wave，提升 16%。但在大 grid size 时，过多的 global memory semaphore 访问成为瓶颈。

术语一般如何实现？如何使用？
TileSync 通过 cuSync 的 policy 模板接口实现（`sem()` 和 `value()` 方法）。cuSyncGen 从 DSL 依赖描述自动生成 TileSync 代码。用户通过 `CuStage<RowMajor, TileSync> stage(grid, tileSize)` 使用。cuSync 内部分配 semaphore 数组：`int* sems = cudaMalloc(grid.x * grid.y * sizeof(int))`。优化变体 TileSync+WRT 额外避免 wait-kernel (W)、重排 tile load (R)、和避免自定义 tile order (T)。

涉及论文标题：
- A Framework for Fine-Grained Synchronization of Dependent GPU Kernels

## cuSyncGen

术语是什么？
cuSyncGen 是 cuSync 框架中的编译器工具，用于从用户描述的 GPU kernel 间 tile 级依赖关系自动生成同步策略（synchronization policies）和 tile 处理顺序（tile processing order）的 CUDA 代码。它包含一个嵌入在 C++ 中的 DSL（领域特定语言），用户在其中定义 kernel 的 grid 维度、tile 的仿射函数依赖关系，cuSyncGen 检查 bounds 正确性后，自动生成：(i) 优化的 tile 处理顺序（最小化 consumer 等待时间）；(ii) 多种同步策略（TileSync、RowSync 及基于依赖结构的变体如 StridedSync）；(iii) 自动应用优化的代码变体。生成结果是可直接插入 cuSync 框架的 CUDA C++ 代码。

从编译框架角度拆解术语：
cuSyncGen 的编译流程（从输入 DSL 到输出 CUDA 代码）：
```
输入: DSL依赖描述（用户编写在C++中）
│
├── Step 1: Define grid dimensions
│   Dim x, y;
│   Grid g1(x, y, H/(2*TileN), B*S/TileN);  // producer grid
│   Grid g2(x, y, H/TileN, B*S/TileN);       // consumer grid
│
├── Step 2: Define tile objects
│   Tile prod(x, y), cons(x, y);
│
├── Step 3: Define tile ranges (ForAll)
│   ForAll prodCols(prod, x, Range(g1.x));
│
├── Step 4: Declare dependencies
│   Dep dep({g2, cons}, {g1, prodCols});
│   // consumer tile(x,y) depends on all producer tiles in same row x
│
▼ 输出: 生成的CUDA代码
│
├── Tile Processing Order (RowMajor):
│   int order(dim2 tile, dim2 grid) {
│       return tile.y * grid.x + tile.x;
│   }
│
├── Policy 1 - TileSync:
│   class TileSync {
│       int sem(dim2 tile, dim2 grid) { return tile.x * grid.y + tile.y; }
│       int value(dim2 tile, dim2 grid) { return 1; }
│   };
│
├── Policy 2 - RowSync:
│   class RowSync {
│       int sem(dim2 tile, dim2 grid) { return tile.y; }
│       int value(dim2 tile, dim2 grid) { return grid.x; }
│   };
│
└── Optimizations (自动判断是否应用):
    - 避免 wait-kernel (2 kernels ≤ 2 waves时)
    - 避免自定义 tile order (同上)
    - 重排 tile load (需要用户用 #pragma tile 标注)
```

cuSyncGen 的 tile 处理顺序生成算法：给定依赖 consumer tile C(x,y) 依赖 N 个 producer tile {P(x, a₀y+b₀), ..., P(x, a_{N-1}y+b_{N-1})}，为实现最小等待时间，将 N 个 producer tile 连续调度：
```
int prodOrder(dim2 tile, dim2 grid) {
    int linear = bid.y * gDim.x + bid.x;
    int y = 0;
    if (tile.y % a₀ <= b₀) y = 0;
    // ... 对其他N-2个tile类似
    else if (tile.y % a_{N-1} <= b_{N-1}) y = N-1;
    return linear / N + y;
}
```
cuSyncGen 自动处理多维依赖——先处理最内层维度（如 x），然后处理外层维度（如 y），为每个维度生成 M∈{1,N} 两种粒度的 policy。对于 Attention 的 strided 依赖，额外生成 StridedSync policy。

术语一般如何实现？如何使用？
cuSyncGen 以 header-only C++ 库形式嵌入 cuSync 项目（开源：github.com/microsoft/cusync）。用户使用流程：(1) 在 C++ 中用 DSL 描述 kernel grid 维度和 tile 依赖；(2) cuSyncGen 编译时 bounds-check 和代码生成；(3) 用户将生成的 policy 类和 order 函数插入 CUDA kernel（通过 `CuStage<Policy, Order>` 模板参数）；(4) 运行所有生成的 policy，选择执行时间最短的。支持 GPU 架构感知的优化（Volta: SM 80, occupancy 等参数可配置）。当前需要用户手动在 kernel 中添加 wait/post 调用和修改 workload 以支持 cuSync，自动化 kernel 修改不在 cuSyncGen 范围内。

涉及论文标题：
- A Framework for Fine-Grained Synchronization of Dependent GPU Kernels

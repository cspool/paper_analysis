## Shepherd Operator

术语是什么？通过联网搜索让回答具体和精准。
Shepherd Operator是Infera compiler对小型DNN operators的合并抽象：当某些operator计算量很小（如elementwise add、activation），单独作为micro operator调度会引入过大的scheduling/fusion/launch overhead，compiler将它们合并为一个子图并创建virtual operator（shepherd operator）。Shepherd operator内部包含多个原始operator的kernel，由shepherd kernel统一管理执行，避免频繁的细粒度调度。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
Shepherd operator在Infera compilation pipeline中的位置：
```
Computation Graph
  ↓ Tile-tailored partition (§4.1)
  ├── Large operator → micro operators (tiled)
  │    e.g., Conv2D → Conv_tile_0, Conv_tile_1, ...
  │    每个tile独立编译为micro-kernel
  └── Small operator group → shepherd operator (merged)
       e.g., Add + ReLU + BatchNorm → Shepherd_0
       内部kernel: k_add, k_relu, k_bn
       shepherd kernel管理调度这些内部kernel
```
设计理由：tile-based调度有overhead（kernel selection metadata、fusion cost、launch latency），对小operator（如单个elementwise op仅几μs执行时间）来说调度overhead可能超过执行时间。Shepherd operator将小operator的调度从per-micro-operator降级为per-shepherd-operator，保持scheduling granularity合理。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Infera compiler在graph partition阶段识别小operator子图（基于operator FLOPs/执行时间估算与调度overhead的比较），合并为shepherd operator。Shepherd kernel类似一个mini-scheduler：在shepherd operator内部按原始operator的拓扑顺序调用子kernel。Shepherd operator对外表现为单一调度单元（一个micro operator），避免TEU SelectKernels/FuseKernels/LaunchKernel pipeline为每个小operator单独调度。论文未给出shepherd operator的具体大小阈值。

涉及论文标题：
- Automated End-to-End Model Serving with Cooperative Compilation and Scheduling

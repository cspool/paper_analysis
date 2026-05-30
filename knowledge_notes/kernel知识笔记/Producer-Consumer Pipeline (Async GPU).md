## Producer-Consumer Pipeline (Async GPU)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Producer-Consumer Pipeline（生产者-消费者流水线）是GPU上利用异步固定功能单元实现计算与数据搬运重叠的软件pipeline技术。在Hopper GPU上，producer（DMA warp通过TMA生产数据到shared memory）和consumer（compute warpgroup通过WGMMA消费shared memory中的数据并产出accumulator）通过named barriers连接，形成深度为PIPE的流水线——producer领先consumer PIPE步预取数据，使TMA的global memory访问延迟被完全隐藏在consumer的计算时间中。

Cypress论文展示的pipeline结构具有三个关键同步原语：
1. prod barriers：TMA完成数据加载后自动arrive，通知consumer数据就绪
2. cons barriers：consumer完成计算后arrive，通知producer该buffer可安全被新数据覆盖
3. copyout barrier：最终output staging完成，通知DMA warp可TMA_store到global memory

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
3-deep Producer-Consumer Pipeline在Hopper GEMM中的执行时间线：

```
PIPE=3, 每个SM执行5个K-reduction迭代

Timeline (时间从左到右):
─────────────────────────────────────────────────────────────────
Iter: k=0       k=1       k=2       k=3       k=4      Epilogue
─────────────────────────────────────────────────────────────────
DMA 0: TMA→s[0]
        Prod[0]████████████████████████████████████
DMA 1:          TMA→s[1]  ██████████████████████████
Comp 0:          wait[0]→WGMMA→cons[0]  █████████
DMA 2:                    TMA→s[2]      █████████████████████
Comp 1:                    wait[1]→WGMMA→cons[1]   ████████
DMA 3:                              TMA→s[0]       ███████████
Comp 2:                              wait[2]→WGMMA→cons[2]███
DMA 4:                                        TMA→s[1]██████
Comp 3:                                        wait→WGMMA→cons
Comp 4:                                               wait→WGMMA→cons
─────────────────────────────────────────────────────────────────
          DMA warp领先Compute PIPE=3步，预取数据覆盖TMA延迟
```

从timeline可见：(1) DMA warp始终比compute领先3步（PIPE步），使TMA异步拷贝的延迟被连续的计算迭代完全隐藏；(2) 只有在第一个PIPE次迭代开始时compute需等待DMA完成（cold start latency），后续迭代因流水线已充满而无额外等待；(3) prod/cons barriers的arrive/wait支持pipeline slot的循环重用（mod PIPE索引）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：
- Pipeline深度选择：通常为2-4——太浅则TMA延迟暴露，太深则shared memory不足或寄存器压力过大
- Shared memory组织：为每个pipeline slot分配独立的buffer（如`sA[T_M, T_K, 3]`——第3维是pipeline depth）
- Barrier数量：每pipeline slot需要一对prod/cons barriers，外加一个copyout barrier
- Backwards anti-dependency：producer在写入buffer slot前必须等待consumer完成上一轮使用该slot的计算——这通过cons barrier实现
- Backwards edge在Cypress IR中显式编码为dashed edges in dependence graph（Figure 12），编译器的pipelining transformation自动插入

涉及论文标题：
- Task-Based Tensor Computations on Modern GPUs
- ThunderKittens: Simple, Fast, and Adorable Kernels

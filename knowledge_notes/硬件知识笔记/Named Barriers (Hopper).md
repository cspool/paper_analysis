## Named Barriers (Hopper)

术语解释
NVIDIA Hopper架构引入的同步原语，允许线程通过名称（ID）而非隐式barrier对象进行同步。Named barriers是实现warp specialization的关键基础设施——DMA warp和compute warpgroup通过命名的prod/cons barriers实现producer-consumer同步。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Named Barriers是Hopper SM内置的硬件同步机制（替代/增强Ampere的隐式barrier）。核心设计：(1) 每个SM提供固定数量的可编程barrier对象（Hopper上通常12-16个），每个由名称/ID索引；(2) 线程通过`arrive`操作将barrier计数器加1，通过`wait`操作阻塞直到barrier达到预期的到达数量；(3) TMA可直接写入barrier——完成数据传输时硬件自动执行arrive，无需consumer线程轮询；(4) 与shared memory（mbarrier）紧耦合——barrier可驻留在shared memory中，允许跨CTA的barrier共享（在TMA multicast场景下）；(5) 支持arrive-on-ed操作——producer设置barrier的期望值，TMA完成后自动触发。

Cypress论文中使用Named Barriers的模式：
```
barrier prod[PIPE], cons[PIPE], copyout

DMA warp:          Compute warpgroup:
  TMA_load(          wait(prod[k%PIPE])
    prod[k%PIPE],    warpgroup_sync()
    ...)             wgmma(...)
                     warpgroup_wait()
                     arrive(cons[k%PIPE])
```

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Named Barrier在Hopper GEMM pipeline中的运转：

```
Pipeline深度PIPE=3，对应3组prod/cons barriers：

时间线（单SM内DMA warp + compute warpgroup）：
  t0: DMA warp发出iteration 0的TMA_load → completion触发arrive(prod[0])
      同时 (异步) 执行iteration 1, 2的TMA_load
  t1: prod[0]到达 → compute warps wait(prod[0])解除
      → warpgroup_sync() → wgmma(iter 0) → warpgroup_wait() → arrive(cons[0])
  t2: cons[0]到达 → DMA warp wait(cons[0])解除（DMA领先compute PIPE步）
      → DMA warp可安全复用sA/sB buffer slot 0
      → TMA_load iteration 3 → arrive(prod[0])  ← 重用barrier 0
  ...
```

Named barrier的关键硬件特性：
- 硬件计数：barrier的arrive/wait操作是单周期硬件原子操作
- TMA集成：TMA completion可直接写入barrier，无中间软件层
- 多播场景：TMA multicast到多个CTA时，每个destination CTA的对应barrier均被自动arrive

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在CUDA编程中使用Named Barriers：
- PTX指令：`mbarrier.init`, `mbarrier.arrive`, `mbarrier.test_wait`, `mbarrier.try_wait`
- CUDA 12.0+提供`cuda::barrier` C++ API（基于mbarrier的实现）
- CUTLASS 3.x在warp-specialized main loop中使用mbarrier for producer-consumer通信
- Named barrier的总数受硬件限制（Hopper: ~12-16 per SM），因此pipeline深度通常为2-4
- 跨CTA mbarrier需要TMA multicast capability

涉及论文标题：
- Task-Based Tensor Computations on Modern GPUs

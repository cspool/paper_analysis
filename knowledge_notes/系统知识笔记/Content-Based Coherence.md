## Content-Based Coherence

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Content-Based Coherence 是 Legion runtime 中维护分布式数据一致性的机制。与基于地址的 coherence（如 cache coherence protocol，通过物理地址确定数据身份）不同，content-based coherence 允许同一数据以多种不同的方式引用（multiple aliasing views），runtime 负责追踪这些 views 之间的关系并在数据被修改时传播更新。例如在 5-point stencil 中，grid 数组有 5 个 aliasing views (center, north, east, west, south)，当 center 被更新时，其他 views 读取相应位置时必须观察到更新后的值。Legion 使用 dynamic dependence analysis [14] 进行精确的依赖分析和数据 coherence 维护。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Content-Based Coherence 在 5-point stencil 中的运转：

```
Grid Array (4×4, 4 GPU partition):
  
  Aliasing Views:
    center = grid[1:3, 1:3]     // 内部 2×2 区域
    north  = grid[0:2, 1:3]     // 上移一行
    east   = grid[1:3, 2:4]     // 右移一列
    west   = grid[1:3, 0:2]     // 左移一列
    south  = grid[2:4, 1:3]     // 下移一行

  Iteration i:
    读取 north, east, west, south, center → 各自 partition
    计算 work, 写入 center 的 partition
    
    Content-based coherence 追踪:
    - GPU 0 写入 center 的 sub-region → 该 sub-region 同时属于
      GPU 2 的 north view、GPU 1 的 east view 等的 sub-region
    - Legion 计算: GPU 0 需要将更新后的 center 数据发送给
      哪些 GPU (使其 aliasing views 获得最新值)
```

对 Diffuse 的关键挑战：aliasing views 使 fusion analysis 复杂化。Diffuse 通过 partition equality check (O(1)) 检测 aliasing——若两个 task 通过不同 partition 访问同一 store，则它们之间的依赖可能跨 processor，融合可能不安全。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Legion 通过 visibility algorithm [14] 实现 content-based coherence——对每个 task 的参数 (region, privilege, fields)，runtime 在 task 提交时计算其必须等待完成的 preceding task 集合。Diffuse 不实现 content-based coherence（委托给 Legion），但它必须在 fusion analysis 中保守处理 aliasing——通过 fusion constraints 的 partition equality check 确保融合后的依赖不破坏 coherence。

涉及论文标题：
- Composing Distributed Computations Through Task and Kernel Fusion

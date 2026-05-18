## IntermediateTensor-based Scheduling（基于中间张量的调度）

术语是什么？通过联网搜索让回答具体和精准。

IntermediateTensor-based Scheduling 是 MetaAttention 提出的 attention kernel 自动调度方法。它将 attention 计算过程中的所有 transient tensor（包括 Q、K、V、scores、state、output 以及 customizable functions 产生的临时张量）统一建模为 IntermediateTensor 对象，每个对象携带三个可配置属性：tile（张量的分块大小）、mem（存储位置——register、shared memory、global memory）、pipelineStage（pipeline 阶段编号，用于 memory copy 与 compute overlap）。调度器通过搜索所有 IntermediateTensor 的最优属性组合来生成执行计划，而非手写固定 tiling/pipeline 策略。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

MetaAttention 的 IntermediateTensor-based scheduling 使用两层调度策略：

**外层：TileConfigScheduling**
```
Func TileConfigScheduling(g: Graph, D: DeviceConfig)
    tiles = EnumerateTiles(g.output_shape, D.basetile)  // 枚举 output tensor 所有可能 tile size
    tensor_tile_graphs = PropagateTileGraphs(g, tiles)  // 将 tile 传播到所有中间张量
    for tile_graph in tensor_tile_graphs:
        plans += TileResourceScheduling(tile_graph, D)  // 内层调度
    for plan in plans:
        if Profile(plan) < best_latency:                // 性能验证
            best_plan = plan
    return best_plan
```

**内层：TileResourceScheduling**
```
Func TileResourceScheduling(g: TileGraph, D: DeviceConfig)
    tensor_list = GetIntermediateTensors(g)
    SetTile(tensor_list, g.tiles)              // 设定 tile size
    SetMem(tensor_list, "L0")                  // 初始全放最高速内存（寄存器）
    tensor_list_sorted = tensor_list.sort(key=lambda t: (len(g[t].use_list), size(t.tile)))
    for tensor_i in tensor_list_sorted:        // 按使用频率×size 降序处理
        plans = EnumerateUnsetAttributes(...)  // 枚举未配置属性（pipeline stage）
        for plan in plans:
            if not MeetMemoryConstraint(plan, D.memoryInfo):
                plans.remove(plan)             // 剔除超内存约束的方案
        if not plans.isEmpty():
            return plans                       // 找到可行方案
        LowerMemLocation(tensor_i.mem)         // 逐步降级 memory tier
    return EmptySet()                          // 无可行方案
```

具体例子：以 Parallel Pattern Softmax Attention 为例，Q/K/V/scores/state/output 以及 RowNorm online 产生的 row_sum/row_max 等都是 IntermediateTensor。scheduler 枚举 output tile 如 [64, 128]，传播到 scores 为 [seq_len_q_tile, seq_len_kv_tile]、Q 为 [seq_len_q_tile, dimqk]、K/V 为 [seq_len_kv_tile, dim]，然后 TileResourceScheduling 尝试将高频使用的 row_sum/row_max 放在寄存器、scores tile 放在 shared memory，按需为 memory copy→compute 分配 pipeline stage，最终生成类似 FlashAttention 的 tiling+pipelining 方案但自动推导。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 MetaAttention 实现中，DeviceConfig 提供硬件约束（basetile 如 H100 128×128、memoryInfo 如 register 65536×32bit、shared memory 228KB），IntermediateTensor 表示每个中间张量的属性。scheduling time 在 H100 上为 46-89 秒（Table 4），短于传统 auto-tuning compiler（如 Ansor）。生成的 scheduling plan 被传递给 attention runtime，由 runtime 将 plan 实例化为具体 kernel（选择 parallel/recurrent template + inline customizable functions + 映射到 TileLang/CUTE 后端）。该方法的关键 trade-off：它不是任意 graph compiler，而是专门针对 attention 的 scheduling——利用 attention 中固定 computation pattern（relevance scoring + aggregation + row-wise norm）缩减搜索空间，从而在分钟级时间内找到高质量执行计划。

涉及论文标题：
- MetaAttention: A Unified and Performant Attention Framework Across Hardware Backends

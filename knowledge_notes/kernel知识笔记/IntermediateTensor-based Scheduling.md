## IntermediateTensor-based Scheduling

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

IntermediateTensor 是 MetaAttention 提出的 attention kernel 调度抽象，将 attention 计算过程中所有中间张量（Q/K/V tiles、scores、weights、output 及 customizable function 内部中间结果）统一建模为带可配置属性的 IntermediateTensor 对象，作为 scheduling space 的基本单元。每个 IntermediateTensor 携带三个属性：(1) **TileShape（tile）**——tensor 的 tile 尺寸，沿 computation graph 传播约束确保相邻 tensors 共享对应维度；(2) **MemoryLocation（mem）**——tensor 在 GPU memory hierarchy 中的位置（Register/Shared Memory/Global Memory），逐级权衡访问延迟与容量；(3) **PipelineStage（pipelineStage）**——tensor 参与的 pipeline 阶段数，如 data copy（TMA async load）和 computation（wgmma MMA）的并发阶段数，决定 buffer 需求和调度灵活性。

该设计的核心洞察：fused attention 融合了多个 operator（matmul + normalization + matmul + elementwise ops），生成大量中间 tensors——其 placement（哪个 memory tier）和 access pattern（何时 load/store、何时可丢弃）直接决定 on-chip memory 利用率和计算延迟。IntermediateTensor 系统化地将这些不直观的调度决策暴露为可枚举、可约束、可优化的离散属性。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

IntermediateTensor 在 scheduling 中的使用流程：
```
// Step 1: 构建 attention computation graph 的 IntermediateTensor 列表
g = ComputationGraph(attention_template)
tensor_list = [
    IntermediateTensor("Q_tile",       tile=?, mem=?, stage=?),
    IntermediateTensor("K_tile",       tile=?, mem=?, stage=?),
    IntermediateTensor("scores_tile",  tile=?, mem=?, stage=?),
    IntermediateTensor("weights_tile", tile=?, mem=?, stage=?),
    IntermediateTensor("V_tile",       tile=?, mem=?, stage=?),
    IntermediateTensor("output_tile",  tile=?, mem=?, stage=?),
    // + customizable function 内部中间 tensors
]

// Step 2: Tile Config Scheduling（外层）——确定所有 tile sizes
tiles_candidates = EnumerateTiles(g.output_shape, D.basetile)
tile_graphs = PropagateTileGraphs(g, tiles_candidates)  // 沿依赖边传播约束

// Step 3: Tile Resource Scheduling（内层）——确定 memory + pipeline
for tile_graph in tile_graphs:
    SetTile(tensor_list, tile_graph.tiles)     // 填充 TileShape
    SetMem(tensor_list, "Register")            // 初始全放最高 tier
    sorted = sort(tensor_list, key=(use_count, size))
    for t in sorted:
        plans = EnumerateUnsetAttributes(t)     // 枚举 pipelineStage
        plans = filter(plans, MeetMemoryConstraint)
        if plans: return plans
        LowerMemLocation(t.mem)                 // Register→SMEM→Global 降级
```

以 Diff-Transformer-3B Softmax Attention (dimqk=128≠dimv=256) 为例：tile propagation 自然地允许 Q_tile[B_r, 128] 和 V_tile[B_c, 256] 的 tile size 不强制对齐（与 FA3 不同，FA3 固定 padding dimqk 到 dimv），避免 compute waste。Memory 分配时 scores_tile（高频使用，use_count=2）优先保留在 Register 或 SMEM，output_tile（低频使用）可降级到 SMEM 或 Global Memory。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MetaAttention 通过两阶段实现：offline scheduling policy 搜索最优 IntermediateTensor 配置（外层枚举 tile sizes ~10-30 候选，内层贪心分配 memory + pipeline ~毫秒级 per candidate），profiling-based 选 latency 最小 plan。搜索结果可按 (attention_config, device) 缓存，跨 run 复用。最终 plan 传入 Attention Runtime，由 kernel template 实例化为具体 CUDA/ROCm kernel（通过 CUTE 或 TileLang backend）。IntermediateTensor 抽象使同一套 scheduling policy 可适配不同 attention variants（因 computation graph 不同而生成不同 intermediate tensor 列表）和不同 hardware（因 DeviceConfig constraint 而限制可行 tile/memory 组合）。

涉及论文标题：
- MetaAttention: A Unified and Performant Attention Framework across Hardware Backends

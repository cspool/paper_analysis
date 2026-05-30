## Two-layer Scheduling Policy for Attention (Tile Config + Tile Resource)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Two-layer Scheduling Policy 是 MetaAttention 提出的分层 attention kernel 调度策略，通过两层递进搜索生成最优 execution plan。外层 **Tile Config Scheduling** 枚举 attention output tensor 的所有合法 tile sizes（受 DeviceConfig.basetile 约束），沿 computation graph 传播 tile shape 到所有 IntermediateTensors（确保相邻 tensors 沿依赖边共享对应维度），生成候选 tile graphs。内层 **Tile Resource Scheduling** 对每个 tile graph 贪心确定所有 IntermediateTensors 的 MemoryLocation（Register/Shared Memory/Global Memory）和 PipelineStage（data copy 与 compute 的重叠阶段数），从最高 memory tier 开始逐步降级直到满足 hardware capacity constraint。所有候选 plans 通过 profiling（或 cost model）选 latency 最优者。

该设计的关键见解：(1) 两层解耦——tile size 决定计算粒度和并行度（影响 compute utilization），但 tile size 组合爆炸；先枚举 tile size（外层搜索空间小，因受 basetile 对齐约束），再对每组 tile size 贪心分配 memory（内层搜索空间更小，因 memory 容量是硬约束）；(2) 传播机制——tile shape 沿 computation graph 传播确保了 fused kernel 中所有 operators 的 tile consistency（如 scores[B_r,B_c] 的 B_r 必须等于 Q[B_r,d] 的 B_r，B_c 必须等于 K[B_c,d] 的 B_c）；(3) 降级策略——从 Register→SMEM→Global Memory 逐级降级，优先保留高频使用 tensor 在高层 memory。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

完整调度流程（以 Softmax Attention, head=32, dim=128, H100, seqlen=2048 为例）：

```
// ====== 外层: Tile Config Scheduling ======
Input: ComputationGraph G (6 IntermediateTensors + customizable functions),
       DeviceConfig D (H100: basetile={64,128}, SMEM=228KB, RF=256KB)

// Step 1: 枚举 output tile sizes
output_shape = [batch*head, seqlen, dimv]  // 实际 batch*head=32
tiles = EnumerateTiles(output_shape, D.basetile)
// 对 dimv=128, 合法 output tiles: Br∈{64,128,256,...}, Bc=128
// → 约 10-30 个候选

// Step 2: 传播 tile shape 到所有 IntermediateTensors
tile_graphs = []
for (Br, Bc) in tiles:
    tile_graph = PropagateTileGraphs(G, (Br, Bc))
    // 传播规则:
    // output[Br, 128] → V_tile: 需 Bc 行 V (dimv=128 共享) → V[Bc, 128]
    // output[Br, 128] → weights[Br, Bc] (aggregation matmul 维度匹配)
    // weights[Br, Bc] → scores[Br, Bc] (row-wise norm 不改变 shape)
    // scores[Br, Bc] → Q[Br, 128] 和 K[Bc, 128] (QK^T matmul 维度匹配)
    tile_graphs.append(tile_graph)

// ====== 内层: Tile Resource Scheduling ======
best_plan, best_latency = None, inf
for tile_graph in tile_graphs:
    plans = TileResourceScheduling(tile_graph, D)
    // 内层流程:
    // 1. SetMem(all_tensors, "Register")  // 初始全放最高 tier
    // 2. sorted = sort(tensors, key=(use_count, tile_size))
    //    高频+大 tile 优先保留在 Register
    // 3. for t in sorted:
    //      plans = EnumeratePipelineStages()
    //         // 对 MMA-heavy: stages=2 (TMA load + wgmma compute overlap)
    //         // 对 memory-bound: stages=1 (无 overlap 但减少 buffer 需求)
    //      valid_plans = filter(plans, MeetMemoryConstraint)
    //         // Σ_tile_sizes ≤ SMEM_228KB, register pressure ≤ 255/thread
    //      if valid_plans: return valid_plans
    //      LowerMemLocation(t)  // Register→SMEM→Global Memory 降级
    for plan in plans:
        latency = Profile(plan)  // 通过 profiling 或 cost model 评估
        if latency < best_latency:
            best_plan, best_latency = plan, latency
```

编译时间：46-89 秒（Table 4），显著快于传统 DL compiler（Ansor: minutes to hours）。
Profiling 开销可通过 cache 消除——相同 (attention_config, device) 组合跨 run 复用 best_plan。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MetaAttention 在 C++/Python 中实现 two-layer scheduling。外层 tile enumeration 通过编译时参数化实现（tile sizes 是模板参数，无需 runtime dispatch）；内层 memory allocation 通过贪心 heuristic 实现（而非 exhaustive search）。Scheduling 结果直接传入 Attention Runtime 的 kernel template selection + code inlining pipeline。跨 hardware 扩展仅需添加 DeviceConfig，无需修改两层 scheduling policy——外层由 basetile 约束搜索空间，内层由 MemoryInfo 约束 resource allocation。

涉及论文标题：
- MetaAttention: A Unified and Performant Attention Framework across Hardware Backends

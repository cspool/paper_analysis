## Intra-operator Communication-Computation Overlap via Tile-Level Fused Kernels（瓦片级通信-计算融合内核的算子内重叠）

术语是什么？
Intra-operator Communication-Computation Overlap 是 MegaScale-MoE 提出的一种细粒度 kernel 级优化技术，将有直接数据依赖关系的通信算子与计算算子以 tile（瓦片）粒度融合到单个 GPU kernel 中执行。与 inter-operator overlap（在不同 CUDA stream 上异步执行独立算子）不同，intra-operator overlap 针对的是通信和计算之间存在直接依赖关系的场景（如 token dispatch 必须在 expert 计算之前完成）。核心思路是将通信和计算的工作负载切分为 tile，使用 device memory barrier（而非 host CPU）实现 tile 级别的细粒度同步——当 remote data tile 到达本地内存时，signal 通知 GEMM kernel 继续计算该 tile。这消除了 host CPU 干预引起的非确定性延迟，也避免了多 stream pipelining 的复杂 stream 控制和尾端计算浪费。

从 kernel 调度角度拆解术语：
MegaScale-MoE 实现了四类 fused kernel（论文 §4.2, Figure 10）：

**类型 1: A2A+GEMM（SP Attention Output Projection）**
```
// All-to-All 通信与 GEMM 计算以 tile 粒度融合
GEMM_Kernel():
  // Step 1: 启动本地数据 tile 的计算 + 远程数据 tile 的通信
  launch_local_GEMM_tiles_on_all_SMs()
  launch_A2A_communication_on_copy_engines()  // 使用 GPU copy engine, 不占用 SM

  // Step 2: 等待远程 tile 到达
  for each remote_tile:
    device_memory_barrier_poll(remote_tile_ready_flag)
    // Flag 由通信完成信号设置
    compute_GEMM_on_tile(remote_tile)

// SM 分配: 少量 SM 处理 A2A 通信管理（数量 tuned 使 comm≈comp latency），
// 其余 SM 全部用于 GEMM 计算
```

**类型 2: GEMM+A2A（SP Attention QKV Projection）**
```
// GEMM 计算完成后，每个 tile 立即发起 remote data transfer
Fused_GEMM_A2A_Kernel():
  for each tile:
    output_tile = GEMM_compute(input_tile, weight)
    // 直接发起 all-to-all remote write（嵌入 kernel 内部）
    A2A_send_async(output_tile, target_rank)
    // 无需额外 kernel launch
```

**类型 3: AG+Scatter+GroupedGEMM（FFN Token Dispatch）**
```
Fused_AG_Scatter_GroupedGEMM_Kernel(input_tokens, routing_map):
  // Step 1: Token 排序以最小化每 tile 的依赖 rank 数
  sorted_tokens = sort_by_expert_then_source_rank(input_tokens, routing_map)
  tiles = slice_into_computation_tiles(sorted_tokens)

  // Step 2: 每个 tile 检查依赖后执行
  for each tile:
    // 等待该 tile 所需的所有 source rank 数据到达
    for rank in tile.dependent_ranks:
      device_memory_barrier_poll(rank_data_ready[rank])
    // Local scatter 内联为按 index mapping 选择输入行（无额外 kernel launch）
    expert_input = select_rows(input_buffer, tile.row_indices)
    // GroupedGEMM 计算
    tile_output = GroupedGEMM(expert_input, expert_weights[tile.expert_id])

// 关键优化:
// - 排序使每个 computation tile 依赖尽可能少的 source rank（理想情况仅 1 个）
// - 减少等待时间，避免重复加载 expert 参数
```

**类型 4: GroupedGEMM+Gather+RS（FFN Token Combine）**
```
// 类型 3 的逆过程：GroupedGEMM 输出 → Gather → Reduce-Scatter
Fused_GroupedGEMM_Gather_RS_Kernel():
  for each tile:
    tile_output = GroupedGEMM(tile_input, expert_weights)
    gathered = local_gather(tile_output, routing_map)
    RS_send_async(gathered, target_rank)
```

**SM 分配与 Swizzling 策略**：
- 对于 A2A+GEMM 类 kernel：分配少量 SM 专门处理通信管理（A2A 比 AG/RS 更复杂），数量通过 profiling tuned 使通信和计算时延匹配
- Swizzling（重排）：重新编排 tile 的通信和计算顺序，使各 rank 的 remote data 到达节奏与 GEMM tile 消费节奏对齐，避免多 rank 同时读写同一 GPU 导致的 NVLink 带宽争用
- 所有同步通过 device memory barrier 实现（类似 FLUX [5]、Comet [53]、TileLink [56] 的方法），无需 host CPU 介入

术语一般如何实现？
- 基于 CUDA 编程模型：使用 device memory（global memory）中的 flag 变量作为 barrier，GPU copy engine 处理数据传输（不占用 SM）
- 依赖项：NCCL 通信库 + CUDA Toolkit + GPU copy engine（NVIDIA GPU 的专用 DMA 引擎）
- 与 inter-operator overlap 的互补关系：
  - Inter-operator overlap：处理无依赖的通信和计算（如 backward 中 activation recomputation 与 gradient communication 并发）
  - Intra-operator overlap：处理有依赖的通信和计算（如 forward 中的 token dispatch 与 expert computation）
- 实测效果：六种 MoE 模型下 fused vs non-fused 通信+计算总时间减少 1.2-4.7x，端到端训练 iteration time 减少 7.1-12.9%
- 论文未开源（ByteDance 内部系统），但技术原理与公开工作 FLUX、Comet、TileLink 一致

涉及论文标题：
- MegaScale-MoE: Large-Scale Communication-Efficient Training of Mixture-of-Experts Models in Production

## Hardware-Aligned Tile Search (WELDER)

术语是什么？
Hardware-Aligned Tile Search是WELDER在枚举tile shape时施加硬件约束的penalty机制。在SubGraphTiling中，EnumerateSubtiles从size=1开始扩展tile shape（类似Roller的tile shape expanding approach），但WELDER在MemTraffic cost model基础上附加三个hardware-aware penalty：(1) **Uncoalesced Memory Access Penalty**——若tile shape导致非连续128B transaction的global memory access，按实际需要的transaction数计算额外traffic；(2) **Parallelism Penalty**——若tile shape太大导致硬件并行度不足（如V100上tile数 < 128），按core utilization比例增加traffic；(3) **Capacity Penalty**——若MemFootprint > target memory capacity，施加infinite penalty直接淘汰。

从kernel调度角度拆解术语：
EnumerateSubtiles with hardware penalties：

```
EnumerateSubtiles(graph, last_config):
  // 从最小的合法tile shape开始
  init_tile = {axis: 1 for axis in output_axes}
  
  for axis in output_axes:
    tile = init_tile.copy()
    while tile[axis] < tensor_dim[axis]:
      // 扩展当前轴的tile size
      tile[axis] = expand_toward_hardware_alignment(tile[axis])
      
      // Penalty 1: Uncoalesced access check
      // CUDA GPU: 128B per transaction (32 FP32 elements)
      if not is_coalesced(tile, transaction_width=128B):
        extra_traffic = calculate_extra_transactions(tile)
      
      // Penalty 2: Parallelism check
      // V100: 80 SMs × 4 warp schedulers × 32 threads = 128 min parallelism
      num_parallel_tiles = total_elements / tile_size
      if num_parallel_tiles < hardware_parallelism:
        extra_traffic *= (hardware_parallelism / num_parallel_tiles)
      
      // Penalty 3: Capacity check
      footprint = MemFootprint(graph_with_tile_config)
      if footprint > target_memory_capacity:
        continue  // skip, infinite penalty
      
      adjusted_traffic = MemTraffic(graph) + extra_traffic
      configs.push(tile_config, priority=adjusted_traffic)
    
  // TensorCore constraints
  for axis marked as MMA_axis:
    // M, N, K must be multiples of MMA fragment size (e.g., 16 for FP16)
    enforce_tile[axis] % MMA_fragment_size == 0
  
  return configs  // sorted by adjusted_traffic ascending
```

V100硬件参数：
- Transaction width: 128B (32 × FP32 elements)
- Min parallelism: 128 (80 SMs × budget for multi-SM occupancy)
- Max block size: 1024 threads
- MMA fragment: 16×16×16 (FP16 TensorCore mma.m16n8k16)

术语一般如何实现？如何使用？
在WELDER的SubGraphTiling中，penalty直接加到MemTraffic上作为优先队列的排序键。这确保搜索首先探索硬件友好的tile shape，而非纯traffic最小但硬件不友好的配置。对于TensorCore，额外添加MMA fragment size整除约束。对于Block size，取所有operator的线程数GCD，若<128则设为128，若>1024则取1024。

涉及论文标题：
- Welder Scheduling Deep Learning Memory Access via Tile-graph

---

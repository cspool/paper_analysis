## Multi-Phase Tiling for Heterogeneous Tree Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Multi-Phase Tiling 是 FastTree 中根据 radix tree 不同层级节点特征自适应选择 attention kernel tile size 的优化技术。核心观察：靠近 root 的节点聚合大量 queries（因许多请求共享相同 system prompt 等长前缀），适合大 tile size 以最大化 shared memory 内的 KV 复用和 tensor core 利用；靠近 leaf 的节点 query 少（仅少数请求共享），大 tile size 导致 shared memory 浪费并降低 SM occupancy（因 block 资源需求大），适合小 tile size 以提升 warp-level parallelism 和 instruction latency hiding。Multi-phase tiling 将 tree 按层级分区（phases），每个 phase 使用不同 tile size 的 kernel 变体，仅在 block-level parallelism 足够大时启用（确保各 phase 均能填满 GPU SM）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Multi-phase tiling 的分区策略和 tile size 选择：

```
Input: Radix tree T with levels L1, L2, ..., Lk
       Each level Li has Ni nodes, each node has Qi queries avg

Algorithm:
  phases = []
  for level i in 1..k:
      if avg(Qi) > threshold_large:
          tile_size = 64          // Large tile: max KV reuse
      elif avg(Qi) > threshold_medium:
          tile_size = 32          // Medium tile
      else:
          tile_size = 16          // Small tile: min shared memory waste

      // Check block parallelism sufficiency
      total_blocks = sum over nodes in level i:
          ceil(Q_node / tile_size)
      if total_blocks >= SM_count * max_blocks_per_SM:
          phases.append((level_i_nodes, tile_size))
      // else: merge with adjacent level

  // Launch separate kernel per phase with its tile size
  for each (nodes, tile_sz) in phases:
      launch_kernel(nodes, tile_size=tile_sz)
```

Effect on SM occupancy:
```
Tile size=64: 每 block 需 (64×128 + 64×128 + 64×128)×2 bytes ≈ 48KB shared mem
              → max 2 blocks/SM (H100, 228KB shared mem/SM)
              → 适合 root level，但 occupancy 低（2×32 warps = 64 warps/SM）
Tile size=16: 每 block 需 (16×128 + 64×128 + 16×128)×2 ≈ 24KB shared mem
              → max 4 blocks/SM
              → 适合 leaf level，occupancy 高（4×16 warps = 64 warps/SM）
```

对比 DeFT（concurrent work）使用 fixed tile size，在 GQA ratio=16 时导致大量 shared memory waste，FastTree 的 multi-phase 设计在此配置下优势明显。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FastTree 的 multi-phase tiling 通过编译多个 tile size 的 Triton kernel 变体实现（如 tile_16, tile_32, tile_64），tree structure-adaptive runtime 在生成 grouping plan 后决定每个 phase 使用的 kernel 变体。分区策略当前实现为简单的水平切分（按 tree level），未来可扩展为更细粒度的 per-node 决策。使用条件：仅当各 phase 的 block parallelism 足够大时启用（否则所有 nodes 使用同一 kernel 变体以最大化 parallelism）。该优化在 tree structure 复杂且 query 分布不均匀时最有效（如 near-root 很多 queries、near-leaf 很少 queries 的场景）。

涉及论文标题：
- FastTree Optimizing Attention Kernel and Runtime for Tree-Structured LLM Inference

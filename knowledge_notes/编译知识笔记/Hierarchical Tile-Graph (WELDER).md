## Hierarchical Tile-Graph (WELDER)

术语是什么？
Hierarchical Tile-Graph是WELDER生成的优化执行计划：从最底层递归分裂为多个sub-graph的多层tile-graph。L0包含完整DNN graph，L1按连接关系分裂为sub-graph，L2进一步分裂。ExecuteGraph递归展开，通过Allocate/LoadTiles/ComputeTile/StoreTiles四条硬件原语生成完整可执行代码。

从编译框架角度拆解术语：
```python
def ExecuteGraph(g, level, in, out):
    mem = Allocate(g.MemFootprint(), level)
    LoadTiles(in, mem)
    for node in g.nodes():
        if level == top: ComputeTile(node, ...)
        else: ExecuteGraph(node.TileGraph(), level+1, ...)
    StoreTiles(mem, out)
```

术语一般如何实现？如何使用？
支持CUDA GPU (global/shared/register三层)、ROCm GPU、GraphCore IPU。扩展host memory层支持超大输入（UNet 8k×8k，通过tile-based streaming跨host+device memory）。codegen通过Load/Store Rewriting（TIR pass将global access改写为shared memory access）和Block/ThreadIdx Remapping完成。

涉及论文标题：
- Welder Scheduling Deep Learning Memory Access via Tile-graph

---

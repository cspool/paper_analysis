## MemTraffic / MemFootprint Cost Model (WELDER)

术语是什么？
MemTraffic和MemFootprint是WELDER的两个解析cost model。**MemTraffic(TileGraph g)**: 计算total traffic = Σ(input_tile_sizes + output_tile_sizes) × num_tile_graphs，intermediate tile在higher level复用不计入traffic。**MemFootprint(TileGraph g)**: bestfit算法按拓扑序计算所有同时存活tile的最大size，超capacity则淘汰该配置（infinite penalty）。两个cost model使WELDER无需硬件profiling即可评估绝大多数候选配置。

从编译框架角度拆解术语：
MemTraffic示例：Matmul(A[98304×64],B[64×128])→Softmax, output tile [16×128]: single-tile traffic = (16×64+64×128+16×128)×4B = 44KB, num_tiles = (98304×128)/(16×128) = 6144, total = 264MB. vs output tile [4×128]: 35KB × 24576 = 840MB. MemTraffic驱动WELDER选择[16×128]（节省69% traffic）。Hardware-Aligned penalty修正：uncoalesced access加额外traffic、并行不足按core utilization比例加penalty。

术语一般如何实现？如何使用？
在SubGraphTiling中，EnumerateSubtiles朝减少MemTraffic方向搜索tile shape（从size=1扩展），MemFootprint确保capacity约束。两个cost model作为优先队列排序键选择top-K最少traffic配置。

涉及论文标题：
- Welder Scheduling Deep Learning Memory Access via Tile-graph

---

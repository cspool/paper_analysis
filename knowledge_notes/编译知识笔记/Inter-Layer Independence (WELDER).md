## Inter-Layer Independence (WELDER)

术语是什么？
Inter-Layer Independence是WELDER解耦优化空间的核心理论性质：某一memory layer的memory traffic仅由该层的tile配置决定，与上层或下层无关。Total traffic = Σ(input_tile_sizes + output_tile_sizes) × num_tile_graphs，仅依赖当前layer tile shape。推论Intra-Layer Independence：同层不同sub-graph的traffic也独立。

从编译框架角度拆解术语：
三层memory (L2:DRAM→L1:SharedMem→L0:Register)：L0优化Conv+ReLU register tile → 最小化L1→L0 traffic；L1优化(Conv+ReLU)+MaxPool shared memory tile → 最小化L2→L1 traffic。改变L0配置不影响L1 traffic公式 → 两层独立优化。

术语一般如何实现？如何使用？
该性质使WELDER递归调度：每层独立搜索 → SubGraphTiling对top-K递归上层 → 不同sub-graph/tile配置并行编译评估。前提：各层容量足够（Propagate后MemFootprint检查）。

涉及论文标题：
- Welder Scheduling Deep Learning Memory Access via Tile-graph

---

## Octree-Search Engine（OSE，Octree 搜索引擎）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Octree-Search Engine（OSE）是 L-PCN 中执行 Octree 查询的硬件引擎：基于 Morton code 定位、采用 linked-list traversal（链表遍历）模式在八叉树上做搜索。L-PCN 的 Partitioning Module 与 Overlap Detection Module 各部署两个 OSE（利用 Octree Buffer 的双端口 BRAM 同时加载两条查询），并行处理两条 Octree-search 查询。Partitioning Module 的 OSE 在 Sampled Octree 上执行邻接节点收集（逐轮向外扩展相邻 voxel 节点），Overlap Detection Module 的 OSE 在 Hub Octree 上对 non-Hub subset 的每个点做存在性查询（命中=重叠、未命中=非重叠）。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
  - L-PCN 中 OSE 的运转流程（以 Overlap Detection 为例，Figure 12）：
```
# 输入: 新 non-Hub subset 的 32 点 + Hub Octree（存于 Hub-Octree Buffer）
OSE_A / OSE_B 并行（双端口 BRAM）:
    for p in subset:
        # Morton code 定位 -> linked-list traversal 下树
        node = Root
        while not leaf:
            node = child[node][morton_bit(node, p)]   # 链表指针遍历
        hit = (node 含 p)
        OverlapIndexes[p] = hit ? OVERLAP : NON_OVERLAP
# 输出: Overlap Indexes -> 决定从 Hub Cache 取结果 or 取特征进 MLP
```
  - 两个 OSE 并行是吞吐关键：Partitioning 与 Overlap Detection 两模块共用同一引擎设计（两个流水化 OSE），支撑运行时岛化与重叠检测的实时性，使 Islandization Unit 延迟开销 <1%。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：FPGA 上为流水化硬件模块，含 Traversal Module（linked-list traversal，Madeira 等 GPU Octrees and optimized search [33]）与 Morton code 计算逻辑；Octree 数据存于分层 BRAM（Octree Buffer / Hub-Octree Buffer）。一般参考：ParallelNN（HPCA'23）是并行 Octree 最近邻搜索加速器；Tigris（MICRO'19）用 Octree 支持 3D 感知。论文未提供 OSE 的公开 RTL。

涉及论文标题：
- L-PCN: A Point Cloud Accelerator Exploiting Spatial Locality through Octree-based Islandization

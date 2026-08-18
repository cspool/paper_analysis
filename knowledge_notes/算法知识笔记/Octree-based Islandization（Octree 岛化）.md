## Octree-based Islandization（Octree 岛化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Octree-based Islandization 是 L-PCN 提出的点云划分方法：利用 Octree（八叉树，一种把 3D 空间递归细分为立方体 voxel 的空间数据结构）的邻接搜索，把 DS 步骤收集到的相邻 point subsets 聚类成"L-PCN Islands"——组内 point subsets 空间强相关（共享大量重叠点）。关键性质：一个 point subset 只能属于一个 Island。该方法分四步：(1) 从 Sampled Point Cloud 选若干中心点为 Hub points；(2) 对每个 Hub point 用 Octree 搜索逐轮收集相邻中心点成 Hub List（重复收集到的节点只保留给最近的 Hub point）；(3) 回原 Input Point Cloud 按 Hub List 把 point subsets 聚成 Islands；(4) 用 Island List 表示每个 Island，供后续 Hub-based Scheduling 使用。该方法与 GCN 加速器 I-GCN（MICRO'21）的 islandization 思想同源（把强内部连接、只与 hub 相连的节点簇聚类以提升片上局部性），但 L-PCN 面向点云的空间稀疏性与无显式邻居索引，改用 Octree 邻接收集。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - L-PCN 论文中 Octree-based Islandization 的伪代码（基于 Sampled Octree 与 Sampled Point Cloud）：
```
# Octree-based Islandization（在 DSU 之后、FC 之前执行）
Input: Sampled Point Cloud（中心点集）; Sampled Octree（剪枝后的八叉树）
# Step1: 随机选固定数量中心点作为 Hub points
Hubs = RandomPick(CentralPoints)
# Step2: 对每个 Hub 用 Octree 搜索逐轮收集邻接中心点
for h in Hubs:
    hubList[h] = [h]
    while 存在未入任何 hubList 的中心点:
        # 每轮沿 Octree 向外扩展一圈相邻 voxel 节点
        nodes = OctreeSearch_Adjacent(SampledOctree, h, round++)
        hubList[h] += nodes 内的中心点
        # 若某 Octree 节点被多个 Hub List 重复收集，中心点只保留给最近 Hub（早轮收集视为更近）
# Step3: 回原 Input Point Cloud 形成 Islands（中心点同属一个 Hub List 的 point subsets 归为一个 Island）
# Step4: 每个 Island 用 Island List 表示（Hub point subset 在首行）
```
  - 时间复杂度受益于 Octree 搜索（树遍历，避免暴力搜索）；与 FPS 造成的"相邻迭代子集空间相距远"问题互补：岛化重排处理粒度，把空间相邻的高重叠子集聚在一起处理。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：L-PCN 在 Islandization Unit 的 Partitioning Module 中硬件实现，两个 Octree-Search Engine（OSE）基于 Morton code + linked-list traversal 在 Sampled Octree 上并行执行邻接节点收集，Sampled Octree 存于分层 BRAM 的 Octree Buffer（双端口）。使用：作为 DSU 与 FCU 之间的插拔模块，兼容准确型与近似型现有 PCN 加速器；论文未提供开源实现。一般参考：Octree 由现有方法预构建（如 ParallelNN HPCA'23 [6]）；I-GCN 的 islandization 开源参考（https://github.com/panmn/I-GCN，MICRO'21）。

涉及论文标题：
- L-PCN: A Point Cloud Accelerator Exploiting Spatial Locality through Octree-based Islandization

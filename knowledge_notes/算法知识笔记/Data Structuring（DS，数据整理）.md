## Data Structuring（DS，数据整理）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Data Structuring（DS）是 point-based 点云网络（PCN）每个基本 Building Block 中的第一步，作用是把空间稀疏、无序的点云整理成"规则"的输入特征图供后续特征计算（FC）使用。一个点云表示为 x = {(p^n, f^n)}，p^n=(x,y,z) 是 3D 坐标，f^n 是特征向量。DS 先选取若干中心点（central points），再对每个中心点做邻居收集（KNN 或 Ball Query）形成 K 个点的 point subset，随后把这些点的特征向量从内存取出来组成输入特征图。DS 是 PCN 独有的操作，无法直接由商用 DLA（如 NPU）加速，未加速时是 PCN 的主要瓶颈；域专用 PCN 加速器主要就是定制 DS 单元（准确型如 PointACC/HgPCN，近似型如 EdgePC/Crescent）。L-PCN 中 DS 由 Data Structuring Unit（DSU）执行，含 Sampling Module（选中心点）、Neighbor Search Module（邻居收集）、Pruning Module（剪枝预构建的 Input Octree 得 Sampled Octree 与 Hub Octrees）。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - L-PCN 论文（PointNet++ Set Abstraction，K=32，1024 点输入）中 DS 的 pipeline 伪代码：
```
# DS pipeline（PointNet++/DGCNN 的 Building Block 第一步）
Input: 点云 X = {(p^n, f^n)}, n=1..N; 预构建 Input Octree
# 1. Sampling：选中心点（如 FPS 从 1024 点选 512 中心点）
C = Sample(X)                      # 中心点集合，|C|≈N/2
# 2. Neighbor Gathering：对每个中心点 c∈C 收集 K 个最近邻
for c in C:
    subset[c] = KNN_or_BallQuery(X, c, K)   # 形成 32 点 subset
# 3. Feature Fetch：从内存取每个 subset 中点的特征向量 f
#    -> 相邻 subset 间共享的重叠点被重复取（冗余访存根源）
Fmap[c] = FetchFeatures(subset[c]) # (K, d_in) 输入特征图
```
  - 关键特征：相邻 point subsets 之间共享大量重叠点（论文基准测量可达 87.5%–93.75% 的重叠率），重叠点的特征被重复从内存取、重复进入后续 MLP——这是 L-PCN 要消除的冗余来源。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 通用实现：GPU/CPU 上用 KNN（如 k-d tree / 暴力搜索）或 Ball Query（固定半径搜索）的 gather kernel；域专用加速器用定制硬件单元，如 PointACC 用 16 个并行距离计算器 + 32-way bitonic sorter 做硬件排名核，HgPCN 用 Octree 缩小搜索空间后排名收集，EdgePC 用 Morton code 索引法近似收集，Crescent 用 KD-tree 近似搜索。L-PCN 假设 Input Octree 由现有方法预构建，DSU 通过 Octree 搜索与剪枝（Pruning Module）得到 Sampled Octree 与 Hub Octrees，供后续岛化与重叠检测复用。论文未提供 DS 的软件实现代码；开源参考：PointNet++ 官方实现（https://github.com/charlesq34/pointnet2）的 farthest_point_sample 与 ball_query 算子。

涉及论文标题：
- L-PCN: A Point Cloud Accelerator Exploiting Spatial Locality through Octree-based Islandization

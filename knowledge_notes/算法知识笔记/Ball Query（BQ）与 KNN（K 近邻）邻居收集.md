## Ball Query（BQ）与 KNN（K 近邻）邻居收集

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Ball Query（球查询）与 KNN（K-Nearest Neighbors）是点云网络 DS 步骤中为每个中心点收集邻居点形成 point subset 的两种标准算法。KNN 取与中心点欧氏距离最近的 K 个点；Ball Query 取以中心点为球心、固定半径 r 的球内所有点（最多 K 个，不足则 padding）。Ball Query 保证固定空间尺度（所有点都在恒定半径内），使学习的局部特征对不同点密度更可泛化；KNN 可能拉到距离差异很大的点。PointNet++ 用 Ball Query，DGCNN 用 KNN。L-PCN 论文以 K=32 为例：每个中心点收集 32 个最近邻形成 point subset，相邻 subset 之间共享大量重叠点（论文实测相邻 subset 重叠可达 87.5%–93.75%）。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - PointNet++/DGCNN 中邻居收集伪代码：
```
# Ball Query（PointNet++）
for c in C:
    subset[c] = { p ∈ X : ||p - c|| <= r }   # 球内点，最多 K 个
# KNN（DGCNN 的 k-NN graph）
for c in C:
    subset[c] = argsort_k(||p - c||_2 for p in X)[:K]  # 最近 K 点
```
  - 在 L-PCN 中，邻居收集由 DSU 的 Neighbor Search Module 执行（准确型如 PointACC 的硬件排名核、近似型如 EdgePC 的 Morton 索引法/Crescent 的 KD-tree 搜索）；收集结果送入 Islandization Unit 做重叠检测与复用。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 通用实现：GPU 上用 PyTorch 的 ball_query / knn_point 算子（PointNet++ 官方仓库 https://github.com/charlesq34/pointnet2）；加速器用并行距离计算器 + 排序器（PointACC 的 16 并行 distance calculator + 32-way bitonic sorter）。L-PCN 的 Islandization 与 Hub-based Scheduling 依赖邻居收集的结果做空间聚类，论文未提供自定义邻居收集的代码。
  - **NS-FPS 补充（ISCA'26）**——NS-FPS 把邻居搜索（neighbor search）从"给每个中心点收集邻域点"升级为 FPS 自身的核心计算原语：利用 Voronoi 图证明距离缓存更新等价于"找落在新采样点搜索球内的邻居"（Eq.5 等价关系），从而把 FPS 重述为迭代邻居搜索。论文按复杂度把邻居搜索分为三类：蛮力搜索（每查询 O(N)，GPU 可并行但线性复杂度不实用）、空间划分（grid/voxel，每 cell O(N/g)，对偏斜分布退化）、层次树（k-d/octree，O(log N) 但不规则访存与遍历开销、动态点云建树代价高）。NS-FPS 的平衡方案：Morton cube 划分 + 自适应半径球查询——每轮只枚举与搜索球相交的 cube（用索引表取点），半径 d_k 随采样推进自适应收缩（120k 帧从覆盖大量 cube 快速降到 <1m），显著减少每轮球查询范围。相关邻居搜索加速器：QuickNN、KD Bonsai（ISA 扩展压缩 k-d 树）、ParallelNN（并行 octree）、Tigris、CAMPER，NS-FPS 借鉴 octree 思路但用 Morton 码查找替代显式建树。

涉及论文标题：
- L-PCN: A Point Cloud Accelerator Exploiting Spatial Locality through Octree-based Islandization
- NS-FPS: Accelerating Farthest Point Sampling via Neighbor Search in Large-Scale Point Clouds

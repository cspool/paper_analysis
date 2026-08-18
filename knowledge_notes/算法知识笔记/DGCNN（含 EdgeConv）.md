## DGCNN（含 EdgeConv）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- DGCNN（Dynamic Graph CNN，Wang et al., TOG 2019）是点云分类/分割网络，核心是 EdgeConv 层：对每个点用 KNN 在特征空间构造动态 k-NN 图，对每条边计算边特征（中心点特征与邻居点特征之差拼接），再经共享 MLP 与 max 聚合更新点特征。由于图在每个 EdgeConv 层动态重建，DGCNN 能捕获局部几何结构。EdgeConv 层即 L-PCN 论文所称 PCN Building Block 的另一实例（用 KNN 而非 Ball Query 收集邻居）。论文用 DGCNN(c)（分类，ModelNet40）与 DGCNN(s)（语义分割，ScanNet）作为 benchmark，并在 DGCNN(c) 上观察到"激活只在本 Building Block 末尾应用时 CONV(A−B)=CONV(A)−CONV(B) 严格成立，可完全补偿结果增量"的特殊情形。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - DGCNN EdgeConv 层 pipeline（Building Block 实例）：
```
# EdgeConv = DS（KNN 动态构图）+ FC（边特征 MLP + max 聚合）
# DS:  对每点 p，KNN 取 K 个邻居构成边 (p, q_i)
#      （相邻点的邻域共享大量重叠邻居 -> 冗余访存与计算）
# FC:  edge_feat_i = MLP([p_feat || (q_i_feat - p_feat)])   # 边特征
#      p_new = max_i(edge_feat_i)                          # 聚合
#      （重叠邻居点重复参与 MLP 与聚合 -> 冗余）
```
  - L-PCN 对 DGCNN 的加速：与 PointNet++ 相同，用 Islandization Unit 聚类 + Hub Cache 复用消除重叠点冗余；理论上 feature fetching 与 feature computation 的削减同样适用。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 通用实现：官方 PyTorch 实现（https://github.com/WangYueFt/dgcnn），用 knn 构建动态图；加速器实现：DGCNN 是常见 PCN 加速器 benchmark（如 PointACC、EdgePC、Mesorasi、L-PCN）。L-PCN 论文未提供 DGCNN 自定义实现，沿用公开模型定义。

涉及论文标题：
- L-PCN: A Point Cloud Accelerator Exploiting Spatial Locality through Octree-based Islandization

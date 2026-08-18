## PointNet++（含 Set Abstraction）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- PointNet++ 是 Qi 等人（NeurIPS 2017）提出的分层点云特征学习网络，解决原始 PointNet 缺乏局部结构捕获能力的缺陷。它由多个 Set Abstraction（SA）层堆叠而成，每层：Sampling Layer（FPS 选中心点）→ Grouping Layer（Ball Query 收集局部邻域）→ PointNet Layer（共享 MLP + max pooling 把邻域聚合为中心点特征）。随着层加深，点数减少、感受野半径与特征维度增大，形成类似 CNN 的分层表示。SA 层即 L-PCN 论文所称 PCN Building Block 的实例。论文用 PointNet++ 三个变体做 benchmark：PointNet++(c)（分类，ModelNet40）、PointNet++(ps)（部件分割，ShapeNet）、PointNet++(s)（语义分割，S3DIS）；其前两个 Set Abstraction 占整体运行时间 90%+（论文 Figure 4 基准）。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - PointNet++ SA 层 pipeline（L-PCN 论文即以此为 Building Block 实例）：
```
# Set Abstraction 层 = DS（采样+分组）+ FC（PointNet MLP + 池化）
# DS:  FPS 选 N' 中心点 -> Ball Query 每组 K=32 邻居 -> 取特征
#      （相邻组共享 ~90% 重叠点 -> 冗余访存）
# FC:  共享 MLP 对每组 32 点计算 (32,6)->(32,128) -> max pool -> (1,128) 中心点特征
#      （重叠点重复进 MLP -> 冗余计算）
# 归一化: 非中心点 XYZ 减去中心点 XYZ（相对坐标），
#         -> 这是 L-PCN 需要 Result Delta Compensation 的原因
```
  - L-PCN 对 SA 的加速点：在 DS 与 FC 之间插入 Islandization Unit，把高重叠相邻子集聚类成 Island 并按 Hub-based Scheduling 复用缓存结果，理论 feature fetching 减 55.2%–90.2%、feature computation 减 45.4%–73.1%（PointNet++/DGCNN 基准）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 通用实现：官方 PyTorch 实现（https://github.com/charlesq34/pointnet2 与 yanx27/Pointnet_Pointnet2_pytorch），含 farthest_point_sample、query_ball_point 与 shared MLP；加速器实现：PointACC、HgPCN、L-PCN 等。L-PCN 用 PointNet++(c) 作为主要原型（DSU 采用 PointACC 的 Mapping Unit、FCU 16×16 脉动阵列）做资源与 cycle-accurate 延迟评估。

涉及论文标题：
- L-PCN: A Point Cloud Accelerator Exploiting Spatial Locality through Octree-based Islandization

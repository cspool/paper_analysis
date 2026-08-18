## Point Feature Table（PFT，点特征表）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Point Feature Table（PFT）是 Mesorasi（MICRO 2020，Feng 等）为点云分析加速提出的数据复用缓冲：存放每个点的两组 MLP 预计算结果——MLP(X,Y,Z,f1,…,fn)（带完整特征）与 MLP(X,Y,Z,0,…,0)（坐标置 0 的中心点特征）。处理每个 32 点 subset 时，从 PFT 取出 32 个完整特征结果与 1 个中心点特征结果，拼接出该 subset 的 MLP(X−Xc,Y−Yc,Z−Zc,f1,…,fn) 近似结果（Delayed-Aggregation 利用了 MLP(A−B) ≈ MLP(A)−MLP(B) 的近似分配律）。PFT 是理解 L-PCN 对照实验的关键：L-PCN 论文指出 Mesorasi 的 Delayed-Aggregation 把负担转移给 PFT——PFT 的密集访存成为新瓶颈，且 MLP 计算与 Delayed-Aggregation 访存被分为两个串行化阶段、无法重叠；当 PFT 超片上容量时片外访存成为关键瓶颈（off-chip 设置下 Mesorasi 明显退化，而 L-PCN 通过运行时缓存复用可重叠访存与计算）。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
  - Mesorasi 中 PFT 的运转流程（L-PCN 论文 VI-C 的描述）：
```
# 预处理阶段（一次）:
PFT[p] = ( MLP(X,Y,Z,f1..fn), MLP(X,Y,Z,0..0) ) for every point p
# 每个 32-point subset 处理阶段:
for subset (中心 c):
    取 PFT 中 32 点的完整特征结果 + 中心点 c 的置零特征结果
    out = 拼接 -> MLP(X-Xc, Y-Yc, Z-Zc, f1..fn) 近似结果（聚合到 c）
# 瓶颈: PFT 需存储所有点预计算结果，访存密集且与计算串行 ->
#       on-chip 存不下时 off-chip 访存无法隐藏（L-PCN 对比实验的设置差异）
```
  - 对比 L-PCN：L-PCN 不用大 PFT，而是用 Hub Cache 按岛运行时缓存增量更新的 MLP 结果，访存与计算可重叠，off-chip 设置下仍保持加速。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：Mesorasi 在 16nm 综合的 NPU 上以片上 buffer 实现（3.8% 面积开销，可扩展），需覆盖全部点特征；开源 artifact：https://github.com/horizon-research/efficient-deep-learning-for-point-clouds（Mesorasi 论文提供）。L-PCN 用它作为对照 baseline，并在 FractalCloud（大尺度 PCN 加速器，原本也部署 Mesorasi 的 Delayed-Aggregation）上把其替换为 Islandization Unit 做对比（提速 1.2×–2.1×、平均节能 48.5%）。

涉及论文标题：
- L-PCN: A Point Cloud Accelerator Exploiting Spatial Locality through Octree-based Islandization

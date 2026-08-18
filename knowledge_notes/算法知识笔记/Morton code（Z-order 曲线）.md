## Morton code（Z-order 曲线）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Morton code（莫顿码，也称 Z-order curve / Z 曲线）是一种空间填充曲线编码：把多维坐标（2D/3D）的各位按维度交错（bit interleaving）合并成一个 1D 整数，使空间上相近的点映射到数值上相近的编码，从而把空间局部性转化为一维线性局部性。Morton code 广泛用于四叉树/八叉树索引、GPU 并行建树（按 Morton 排序点即可递归细分）、点云压缩与数据库空间索引。L-PCN 中，Octree-Search Engine（OSE）基于 Morton code 在 Octree 上执行查询（配合 linked-list traversal 遍历模式），两个 OSE 并行处理两条 Octree-search 查询；EdgePC 也用 Morton code 对点结构化后做近似邻居收集。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - 3D Morton code 编码伪代码（每坐标 b bit，总码长 3b bit）：
```
def morton3(x, y, z):              # 逐位交错
    code = 0
    for i in range(b):             # 从低位到高位
        code |= ((x >> i) & 1) << (3*i)
        code |= ((y >> i) & 1) << (3*i+1)
        code |= ((z >> i) & 1) << (3*i+2)
    return code                    # 空间邻近点 -> 码值邻近（Z 形扫描序）
# L-PCN 用法：OSE 用 Morton code 定位 Octree 节点/点，
#   配合 linked-list traversal 在 Sampled Octree / Hub Octree 上搜索
```
  - 在 L-PCN 中的作用：Octree 搜索的索引原语——按 Morton 码对点排序可线性化空间位置，使八叉树节点的子节点地址可由 Morton 码直接推导（常数时间定位），两个 OSE 才能高效并行执行邻接节点收集与重叠检测。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：软件库如 Go 的 github.com/habedi/morton、Rust 的 space crate（支持 BMI2 pdep/pext 加速）；硬件里用按位交叉逻辑或 LUT。使用：GPU 并行建 octree 时先对点算 Morton 码并排序；L-PCN 的 OSE 用 linked-list traversal 模式（Madeira et al. GPU Octrees and optimized search [33]）遍历。论文未提供 OSE 的公开 RTL；一般参考 ParallelNN（HPCA'23）的并行 Octree 搜索加速器。
  - **NS-FPS 补充（ISCA'26）**——NS-FPS 以 Morton 码作为 FPS 邻居搜索的空间索引原语，替代 k-d 树/八叉树建树：坐标量化成 15/15/11-bit 整数后取 7/7/3 个 MSB 按位交织成 17-bit Morton 码（默认配置；敏感性分析比较 (5,5,1)/(6,6,2)/(7,7,3)，32k 点用 (6,6,2) 最优、120k 用 (7,7,3) 最优），每个码隐式定义一个 3D cube。重排用**桶排序**（线性时间，免比较排序/免建树，显著低于 k-d 树的预处理延迟）。邻居查询时枚举与搜索球 B(s_k,d_k) 相交的 cube、用索引表取 cube 内点，滤掉不可能更新的远处点。这与 L-PCN 用 Morton 码做八叉树查询的用法互补：NS-FPS 直接以 Morton 码分组存点、无显式树结构。
  - NS-FPS 中 Morton 编码伪代码（硬件 4 级流水）：p(x,y,z)→量化 (x_q,y_q,z_q)（15/15/11-bit）→取 MSB (x_h,y_h,z_h)（7/7/3-bit）→交织得 17-bit Morton code→查 Occupancy Table（新 cube 建 Page Table 项，否则复用）→分配/追加 Page Memory。GPU 版 NS-FPS 因 Morton 遍历在 GPU 通用内存层级上开销大，小规模劣于 QuickFPS-GPU，故需 ASIC 释放潜力。

涉及论文标题：
- L-PCN: A Point Cloud Accelerator Exploiting Spatial Locality through Octree-based Islandization
- NS-FPS: Accelerating Farthest Point Sampling via Neighbor Search in Large-Scale Point Clouds

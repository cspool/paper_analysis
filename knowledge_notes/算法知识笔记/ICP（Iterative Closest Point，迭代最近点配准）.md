## ICP（Iterative Closest Point，迭代最近点配准）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- ICP 是点云配准（registration）的标准迭代算法：给定两帧点云（如自动驾驶车辆运动前后 LiDAR 扫描的 Q 与 P），通过迭代"找最近邻对应点 → 估计刚体变换（旋转 R/平移 t）→ 应用变换"最小化两帧点云的几何错位。RoboCortex 以 ICP 为端到端评估 workload：每次迭代的核心开销是 NNS（找 P 中每点在 Q 中的 k 近邻），加 Jacobi SVD 分解估计变换。变体按最近邻数与矩阵迭代算法区分：point-to-plane（k=10）、point-to-line（k=5）、point-to-point（k=1，纯最近邻）。k 越小，NNS 优化带来的端到端收益越大。NNS 占单次配准延迟 53.25%-71.66%。
- 从算法pipeline角度拆解术语，给出具体计算过程例子：ICP 一次迭代（Initial Registration）流水 = Build（初始空间数据结构构建，仅首帧）→ NNS（对 P 每点在 Q 中搜 k 近邻）→ SVD（Jacobi SVD 分解估计 R/t）→ 应用变换；Following Registration 只含 NNS+SVD。端到端例子（Fig. 17）：ICP point-to-plane 在 KITTI-360 上，RoboCortex 的 NNS 优化使并行部分 2.27× 提升、端到端 18%-28% 改善；随真实配准进行，收益从 Initial Registration 转移到 Following Registration（Build 一次性开销摊薄）。三种模式对比（Fig. 18，去除数据加载时间）：k 越小 NNS 优化收益越显著。
- 术语一般如何实现？如何使用？：软件实现参考 https://github.com/FeeZhu/ICP（论文引用 [13]）；点云库 PCL 有 pcl::IterativeClosestPoint。硬件/系统使用：RoboCortex 在 zsim 上执行 ICP（对标 Jetson AGX Orin），RSU 加速 NNS、Path Buffer 挖掘物理局部性、语义预取降低缓存 miss；对比 baseline 包括 base CPU、Tartan（机器人 CPU，近似 NNS+预取，收益有限且牺牲精度）、stream 预取器版 RoboCortex。数据集：自主驾驶（KITTI-360）与物体重建（EPFL Statues）。缩放实验（Fig. 20）：1/8/16 核下 RoboCortex 均显著优于 baseline 与 Tartan。

涉及论文标题：
- Optimizing Spatial Data Structure with Near-Cache Acceleration by Exploiting Physical Locality（RoboCortex）

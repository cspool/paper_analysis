## 空间数据结构（Spatial Data Structure：Kd-Tree / Octo-Tree / R-Tree / BVH）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 空间数据结构是组织几何数据、避免线性遍历的基本抽象，用于点云处理、光线追踪、碰撞检测。共同特征：每个非叶节点代表一个物理空间，子节点空间包含于父节点空间，叶节点对应点/图元。三种代表结构按空间划分方式区分：Kd-Tree（Bentley 1975）沿不同维度用平面交替二分，逐层平分点分布（split-domain 顶向下）；Octo-Tree 每个节点 8 个子节点，对应三维边长各半分的 8 个子立方（split-domain）；R-Tree（Guttman 1984）自底向上构建，父节点是全部子节点包围盒（bounding-box）。BVH 是光线追踪的包围体层次结构（同属 bounding-box 类）。NNS 复杂度从暴力 O(NM) 降到 O(N log M)。
- 从算法pipeline角度拆解术语，给出具体计算过程例子：Kd-Tree NNS（Listing 1）三步通用结构：(i) 叶节点处理——算查询点 Pos 与叶内点的距离并更新结果表 ResList；(ii) 扩展规则（结构相关）——比较 Pos[Node.Axis] 与分裂阈值 Node.Thresh 决定先走左/右子节点；(iii) 递归+剪枝（通用）——`if NeedExpand(Pos, Node, ResList): KdTreeNNS(Pos, inf_child, ResList)`，NeedExpand 用 `d*d < ResList.MaxDis()*Alpha` 判定是否回溯展开另一分支。关键区别：与 B+ 树/skip list 等 1D 索引不同，空间数据结构"包含 ≠ 邻近"（Containment ≠ proximity），DFS 回溯不可避免（Fig. 4 反例：坐标小于阈值却仍可能需访问另一分支），这是它难以被哈希/非回溯加速器加速的根本原因。Octo-Tree 的距离计算用 Iter 原语对 8 分支流水化复用计算单元；R-Tree 用优先队列按包围盒距离决定访问顺序。
- 术语一般如何实现？如何使用？：软件实现常用 PCL（Point Cloud Library，RoboCortex 用它构建三种结构）；搜索算法通常带 k-d tree kNN 的 FLANN、CGAL、Embree（BVH 构建）。硬件/系统使用：RoboCortex 用 RSU 数据流（Stack 用于 Kd/Octo-Tree、Priority Queue 用于 R-Tree）执行搜索，不同数据结构共享同一套原语——把搜索拆成叶距离/扩展/递归三部分即可适配。数据集：自主驾驶（KITTI-360，点分散）vs 物体重建（EPFL Statues，点紧凑）；实验结果：Kd-Tree 最受益于物理局部性+预取，Octo-Tree 最受益于 RSU 硬件本身（其地址连续、L1/L2 hit 本就最高），R-Tree 在紧凑点云收益最大。

涉及论文标题：
- Optimizing Spatial Data Structure with Near-Cache Acceleration by Exploiting Physical Locality（RoboCortex）

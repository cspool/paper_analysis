## Physical Locality（物理局部性）与 Belief Space（信念空间）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Physical Locality 指空间数据结构搜索中"空间上相邻的查询点往往共享搜索路径"的现象：点云 P 中相邻点（几何邻近）的最近邻搜索通常经过同一批父节点，即共享搜索前缀。传统缓存只能利用内存地址局部性（spatial/temporal），对物理坐标无感知；RoboCortex 通过 RSU 让缓存获得物理坐标后，物理局部性才可被挖掘。与之对比 L-PCN（ISCA'26，Octree Islandization）也挖掘点云"空间局部性"，但目标是把高度重叠的点子集聚成 island 做数据复用（PCN MLP 推理），RoboCortex 目标是 NNS 搜索路径复用。
- 从硬件架构角度拆解术语，给出运转流程具体例子：Belief Space（信念空间）是保证无精度损失复用共享路径的几何构造：对共享空间 S（某中间节点代表的三维空间）与其中一点 x，存在子区域 B⊂S（1D 构造为 $B=[(x_{inf}+x)/2,(x_{sup}+x)/2]$；2D 为两条抛物线夹的锥形区；3D 推广为 Theorem 1 的判定规则），使得任意 p∈B 的最近邻必落在 S 内。因此硬件只需检查"p 到 x 的距离 < p 到 S 所有边界距离"即可安全地从中间节点开始搜索，避免回溯破坏局部性利用（"Containment ≠ proximity"——节点包含不等于近邻包含，这是空间数据结构与 B+ 树等 1D 索引的本质区别）。运转例子：Kd-Tree 上完成一次 NNS 后记录 S（最深共享父节点包围盒）与 x → 下一相邻查询 p 命中 B → 直接从中间节点搜索 → 跳过根到中间节点的全部节点访问。
- 术语一般如何实现？如何使用？：硬件实现为 Path Buffer 中的距离比较器 + 空间记录；判定规则（Theorem 1）是程序可执行的距离比较而非解析几何求解。使用场景与收益：紧凑点云（物体重建 Statues）物理局部性更强、收益更大（12.73-77.94× vs 自主驾驶 2.74-13.07×）；Kd-Tree/R-Tree 对物理局部性最敏感，Octo-Tree 不敏感。泛化：前提是"查询目标（叶）包含于中间节点"的稳定包含关系，可推广到 BVH 光线追踪（额外 4-34%）。

涉及论文标题：
- Optimizing Spatial Data Structure with Near-Cache Acceleration by Exploiting Physical Locality（RoboCortex）

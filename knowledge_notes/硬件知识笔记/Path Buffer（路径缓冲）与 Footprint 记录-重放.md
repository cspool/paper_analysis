## Path Buffer（路径缓冲）与 Footprint 记录-重放

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Path Buffer 是 RoboCortex 提出的专用硬件结构，缓存"频繁出现的搜索路径"以利用物理局部性：每次 NNS 结束后，记录三元组（共享空间 S——k 个最近邻点的最深共享父节点所代表的空间，最近邻点 x 的坐标，叶节点地址 Addr）。下次搜索点 p 若落在由 x 与 S 边界构造的 belief space B 内（Theorem 1），则保证 p 的最近邻必在 S 内，可直接从中间节点（非根节点）开始搜索，绕过根到该节点的冗余访问。配置：16 条目、LRU 替换（论文测 R-Tree/Kd-Tree 性能随容量近似线性增长、Octo-Tree 不敏感，权衡面积取 16），每条目 40B（INT32 单精度坐标），总开销可控在 ~512B 内（远小于移动 CPU L1 的 32-128KB）。
- 从硬件架构角度拆解术语，给出运转流程具体例子：四阶段协作流程：(i) Path Buffer Lookup——按 p 坐标查 B，命中则把 RootAddr 替换为中间节点地址，否则从根搜索；(ii) RSU Recursive Searching——RSU 数据流执行 DFS，输出优先队列存 k 近邻点及其 footprint（二进制路径记录，1=左/0=右，如节点 1R2L3L 记为 1011）；(iii) Footprint Tracing——从根开始搜索时，用 k 个 footprint 的公共前缀找共享父节点（1011 与 10xx 的前缀 10），再反向右移 footprint 逐位从根走到该中间节点（利用 RSU 数据流重用），避免显式栈溢出；(iv) Path Buffer Updating——LRU 插入新三元组。本质是"用计算换内存"：重搜索开销 <1% 总搜索时间。关键难点：DFS 中为了防栈溢出会弹掉父节点，footprint 正是为事后找回共享父节点而设计。
- 术语一般如何实现？如何使用？：Path Buffer 在 RTL 实现（UMC 28nm 综合 0.344 mm²），与 RSU 协同；判定 B 用 Theorem 1 的距离比较（p 到 x 距离 < p 到 S 各边界距离）。使用场景：物理局部性强的点云（物体重建 > 自主驾驶）、Kd-Tree/R-Tree 收益大；Octo-Tree 因地址连续、本身局部性好而收益小。论文未开源。

涉及论文标题：
- Optimizing Spatial Data Structure with Near-Cache Acceleration by Exploiting Physical Locality（RoboCortex）

## 最近邻搜索（NNS，Nearest Neighbor Search / KNN）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 最近邻搜索是为查询点找到数据集中距离最近的 k 个点（KNN）的基础操作，是点云配准（ICP）、SLAM、碰撞检测的核心开销。点云场景（RoboCortex）：对源点云 P 的每个点 p，在目标点云 Q（N,M>10³）中搜 k 近邻；暴力法 O(NM)，用 Kd-Tree 等空间数据结构可降到 O(N log M)。并行策略：把不同源点绑定不同线程（Listing 2 ConcurrentNNS 的 parallel_for）。注意与 ANNS（近似最近邻，如 Faiss/量化/哈希）区分：RoboCortex 是精确 NNS（无精度损失，这是相对 Tartan 近似算法的卖点）。与 L-PCN/PointNet++ 的 Ball Query/KNN 邻居收集不同：那是为 MLP 特征计算收集点集，RoboCortex 是为 ICP 配准找几何最近邻。
- 从算法pipeline角度拆解术语，给出具体计算过程例子：Kd-Tree KNN 伪代码（RoboCortex Listing 1）：
```
def KdTreeNNS(Pos, Node, ResList):
    if Node.IsLeaf():                       # Part 1 叶节点
        ComputeDisForLeaf(Pos, Node)
    if Pos[Node.Axis] < Node.Thresh:        # Part 2 扩展规则
        sup_child, inf_child = Node.Left, Node.Right
    else:
        sup_child, inf_child = Node.Right, Node.Left
    KdTreeNNS(Pos, sup_child, ResList)      # Part 3 递归
    if NeedExpand(Pos, Node, ResList):
        KdTreeNNS(Pos, inf_child, ResList)
def NeedExpand(Pos, Node, ResList):
    if ResList.size() < k: return True
    d = Pos[Node.Axis] - Node.Thresh
    return d*d < ResList.MaxDis()*Alpha
```
执行链：根节点 → 按维度坐标比较下探 → 到叶算距离更新 ResList（MaxD 用 Reg 维护）→ 回溯时 NeedExpand 判定是否展开另一分支 → 输出 k 近邻。NNS 占 ICP 一次配准延迟的 53.25%（point-to-plane）到 71.66%（point-to-point），是核心瓶颈。
- 术语一般如何实现？如何使用？：软件：PCL kdtree/knn、FLANN、scikit-learn KDTree；硬件/系统：RoboCortex 用 near-cache RSU 数据流加速（显式栈支持递归回溯）+ Path Buffer 物理局部性复用 + RSU 引导预取，NNS 相对 CPU 加速 2.74-13.07×（自主驾驶）/12.73-77.94×（物体重建）。GPU 方案（RTX、RTNN ray tracing）受分支发散与 CPU-GPU 数据搬运（占 16.24-42.57%）限制加速 <2×。精确 NNS 与近似（Tartan）对比：高精度需求下近似方案性能收敛于 baseline，RoboCortex 不牺牲精度。

涉及论文标题：
- Optimizing Spatial Data Structure with Near-Cache Acceleration by Exploiting Physical Locality（RoboCortex）

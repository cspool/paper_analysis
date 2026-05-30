## EP-First vs DP-First Placement (专家优先 vs 数据优先的设备放置策略)

术语是什么？

在大规模 MoE 训练的 GPU 集群中，如何将 EP（Expert Parallelism）和 DP（Data Parallelism）映射到物理设备上存在两种对立策略：

- **EP-First（EP-then-DP）**：优先将不同的 expert 放置在同一节点内，最大化 intra-node expert diversity，然后在节点间复制整组 expert（DP 跨节点）。优点：alltoall 通信主要集中在节点内（快速）；缺点：DP 梯度同步需跨节点（慢速），对大模型参数量大时不利。
- **DP-First（DP-then-EP）**：优先将同一 expert 的副本放在同一节点内，使 DP 梯度同步局限在节点内（快速），expert 分布在跨节点（EP alltoall 跨节点）。优点：DP 通信利用快速 intra-node 链路；缺点：EP alltoall 需跨节点。

选择取决于模型大小和硬件拓扑的带宽不对称程度。Frontier 上 intra-node 200 GB/s vs inter-node 25 GB/s（8:1），大模型时 DP-First 将梯度同步从慢速 inter-node 迁移到快速 intra-node，收益更大。

涉及论文标题：
- X-MoE: Enabling Scalable Training for Emerging Mixture-of-Experts Architectures on HPC Platforms

## Fat-Tree Topology

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fat-Tree（胖树）是一种数据中心网络的 Clos-based 多级交换拓扑，由 Al-Fares et al. (SIGCOMM 2008) 提出。在 Fat-tree 中，交换机分为 edge/leaf（连接 server）、aggregation、core（连接 aggregation）三层。leaf 交换机的上行带宽等于下行带宽的 k/2 倍（k 为交换机端口数），使得 upper layer 链路的带宽累积（"fat"），从而在任意 server 对之间提供 full bisection bandwidth。典型的三层 Fat-tree（k=8）可支持 128 servers。Fat-tree 的主要优势：1:1 non-blocking（所有 server 可同时以线速通信）、无超额订阅（oversubscription ratio = 1:1）、ECMP（Equal-Cost Multi-Path）多路径负载均衡。代价：需要大量交换机和布线的 CAPEX（k=8 需 20 台交换机，k=64 需 5120 台）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 MixNet 中，Fat-tree 用作 EPS 部分的网络拓扑（MixNet 的"电气域"）。每 server 的 2 个 EPS NIC 连接到 Fat-tree leaf 交换机，通过 spine 交换机实现全局互联。

Fat-tree 在 MoE 训练中的问题（来自 MixNet 的生产环境测量）：
- EP 的 all-to-all 通信是 sparse 和动态的——每个 iteration 只有少数 GPU 对之间有大量数据传输（heavy hitters），大部分链路空闲。
- Fat-tree 提供 uniform full bisection bandwidth——所有 leaf-spine 链路的带宽相等，无论实际 traffic 分布如何。这导致过度 provisioning：大多数链路在大多数时间内处于低利用率状态。
- 相比之下，MixNet 的 OCS 高带宽域仅在需要时才为 heavy hitter GPU 对建立直连电路——按需分配带宽，不浪费链路资源。

Fat-tree 的 networking cost 构成（per 100G link, Table 4）：transceiver $99 + NIC $659 + switch port $187。但在 Fat-tree 中，每个 inter-server 连接需经过 2 级交换机（leaf + spine），即每个 inter-server 连接消耗 4 个 switch ports（leaf in, leaf out, spine in, spine out）。MixNet 的 OCS 域：transceiver $99 + NIC $659 + OCS port $520 = ~$1278/link，但仅需 1 个 OCS port per link（直连，无中间层）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 标准部署：使用 BGP/OSPF + ECMP 多路径负载均衡。NVIDIA/Mellanox Spectrum 交换机广泛支持 Fat-tree。
- Rail-optimized 变体：NVIDIA 推荐的 GPU 集群拓扑——将同 rank 的 GPU 连接到同一 leaf 交换机，实现同 rail 内 GPU 的低延迟 NVLink-like 通信。MixNet 中 Rail-optimized 是重要的对比 baseline。
- Oversubscribed Fat-tree：3:1 oversubscription（leaf 上行带宽 = 下行带宽/3），降低成本但引入 congestion（当 heavy hitter 对恰好在 oversubscribed 路径上时）。MixNet 比 3:1 Fat-tree 快 1.6×。
- 相关变体：VL2（valiant load balancing）、Jupiter（Google's Clos fabric）、MegaScale（ByteDance's 3-layer Clos）。

涉及论文标题：
- MixNet: A Runtime Reconfigurable Optical-Electrical Fabric for Distributed Mixture-of-Experts Training

## Dragonfly Network Topology (蜻蜓网络拓扑 / Dragonfly Topology for HPC)

术语是什么？

Dragonfly 拓扑是一种面向高性能计算的层次化网络拓扑结构，由 John Kim 等人于 2008 年提出（ISCA 2008），被 Frontier 等超级计算机采用。其核心设计思想是将计算节点分组为层次化结构：一组节点通过高带宽交换机形成 group（或 rack），groups 之间通过较低带宽的全局链路连接。任意两个 group 之间至少有一条直接链路（dragonfly 的"翅膀"），使网络直径很小（通常 3 跳以内），但全局链路带宽远低于 group 内带宽。

从硬件架构角度拆解：

Dragonfly 在网络架构中的关键特征与 X-MoE 的关系：
- **带宽不对称性**：Group 内带宽 >> Group 间带宽（Frontier 上 200 GB/s vs 25 GB/s ≈ 8:1），要求通信感知的调度策略
- **层次化路由**：最短路径通常为 source→local switch→global link→destination switch→destination（3 跳），但全局链路拥塞时需自适应路由使用更多跳
- **对 MoE alltoall 的影响**：传统 MoE 系统将所有 GPU 等权对待（如 DeepSpeed-MoE），在 Dragonfly 上造成严重的跨 group 通信拥塞。X-MoE 的 RBD 通过将 inter-group 通信限制为仅 pilot tokens、intra-group 通信处理 local replica 来适配这一拓扑特征

X-MoE 在 Frontier Dragonfly 上观察到的现象：
- 冗余率随 EP 规模增大（最高 75.1%），即 75.1% 的跨节点 token 传输是重复的
- Cross-rack（>256 GPU）时 alltoall 延迟异常升高，因全局链路拥塞 + 其他作业竞争

术语一般如何实现？

Dragonfly 拓扑的实现：Frontier 使用 Slingshot-11 交换机构建 Dragonfly，每节点 4 个 Slingshot NIC（25 GB/s each）。X-MoE 通过限制 EP ≤256 使 alltoall 组保持在单 rack 内，并通过 RBD 的 pilot/local replica 分层策略减少跨 group 通信量。

涉及论文标题：
- X-MoE: Enabling Scalable Training for Emerging Mixture-of-Experts Architectures on HPC Platforms

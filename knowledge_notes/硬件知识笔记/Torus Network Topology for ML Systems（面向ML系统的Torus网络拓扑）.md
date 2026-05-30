## Torus Network Topology for ML Systems（面向ML系统的Torus网络拓扑）

术语是什么？
Torus 网络拓扑是一种直接互连网络（direct interconnect），每个节点通过短距链路连接到相邻节点，且每维边界节点通过 wrap-around 链路闭合为环。N-D torus 中每个节点有 2N 条链路（±X, ±Y, ±Z, ...），具有高路径分集性和成本效益扩展性——无需昂贵的高 radix 交换机即可连接大规模 AI 加速器。ML 系统中的典型部署：Google TPUv3 (8×8 2D torus)、TPUv4 (4×4×4 3D torus, 可通过 OCS 扩展到 8×4×4 双 pod)、Fugaku 超算 (Tofu 6D mesh/torus)、Amazon Trainium、Graphcore IPU-POD 等。

从硬件架构角度拆解术语：
以 TPUv4 3D-torus 为例的硬件组织：
1. **物理组织**：TPUv4 芯片在 3D torus 中排列，每个芯片有 6 条 ICI（Inter-Chip Interconnect）链路——±X/Y/Z 方向各一条，每条链路带宽 56 GB/s。4×4×4 pod 共 64 个 TPU，通过 48 个 OCS（Optical Circuit Switch）实现可重构互联。
2. **链路带宽利用**：单维度 All-to-All 时仅使用该维度方向的 2 条链路（±方向），其余 4 条链路闲置。DimRotation 通过轮转维度顺序使各维度链路交错使用，消除闲置。3D torus 上全维度利用率可达 100%。
3. **故障影响**：TPUv4 依赖 OCS 连接 torus 的各维度 wrap-around 链路。OCS 故障会导致包含该 OCS 的所有 wrap-around 链路同时失效（如 OCS 34 故障导致 X-Z 平面 4 条链路失效），形成周期性故障模式。TPUv4 的 ICI 路由通过 dateline 机制避免 torus 环上的死锁。

术语一般如何实现？如何使用？
Torus 网络通常与 DOR（Dimension-Order Routing）配合——DOR 按固定维度顺序（如 XYZ）路由数据，先沿 X-dim 到达目标列，再沿 Y-dim 到目标行，最后沿 Z-dim 到目标芯片。优点是简单无死锁，缺点是固定顺序导致先遍历维度拥塞、后遍历维度闲置。本文提出的 HalfRing+DimRotation 通过逐跳 store-and-forward 编排替代 DOR 的多跳硬件路由，消除拥塞并最大化利用率。在 TPUv4 4×4×4 pod 上，DOR 实测 All-to-All 带宽 75.2-75.9 GB/s，HalfRing+DimRotation 实现 1.57× 加速。

涉及论文标题：
- Optimizing All-to-All Collective Communication with FaultTolerance on Torus Networks

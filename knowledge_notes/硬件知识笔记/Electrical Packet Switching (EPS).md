## Electrical Packet Switching (EPS)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Electrical Packet Switching (EPS，电气包交换) 是当前数据中心网络的主流交换技术。数据被封装为 packet，交换机通过解析 packet header（MAC/IP/TCP 层）进行逐跳存储转发（store-and-forward）或直通转发（cut-through）。EPS 交换机包含：输入端口 buffer、包解析器、路由查表引擎（基于 MAC 表/IP 路由表/flow table）、交换 fabric（crossbar 或 shared memory）、输出端口 buffer 和调度器。EPS 的优势：massive scalability（可扩展到数十万端口）、无连接（connectionless，无需预先配置电路）、原生支持多路复用和统计复用（statistical multiplexing）。代价：per-hop 延迟（包括排队、处理、序列化延迟），功耗随带宽线性增长（SerDes/Retimer 功耗显著），在 MoE 训练中静态拓扑无法适应动态 traffic pattern。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 MixNet 的混合光电架构中，EPS 承担两类通信任务：
1. **全局通信（DP + PP）**：EPS 使用 Fat-tree 拓扑连接所有 GPU server 的 EPS NIC（每 server 2 个），提供全局全互联的网络可达性。DP 的 all-reduce 梯度同步通过 hierarchical all-reduce（intra-host NVSwitch reduce → inter-host EPS ring all-reduce → intra-host NVSwitch broadcast）在 EPS 上执行。PP 的 point-to-point hidden state 传输同样经 EPS。
2. **EP fallback 路径**：当 OCS 端口资源不足以覆盖所有通信密集型 GPU 对时，剩余的 EP all-to-all 流量通过 EPS 传输（作为 OCS 直连的 fallback）。MixNet 的 topology-aware routing 优先选择 OCS 直连电路，仅在 OCS 不可达时才走 EPS。

EPS 在 MixNet 中的运转流程（以 DP all-reduce，1024 GPU 为例）：
1. Intra-host reduce（NVSwitch）：各 server 内 8 GPU 通过 NVSwitch 将梯度聚合到 1 个 gateway GPU（连接 EPS NIC）。
2. Inter-host ring all-reduce（EPS Fat-tree）：gateway GPU 通过 EPS NIC 在 ring 上执行 all-reduce。数据从 GPU 显存 → NIC（DMA）→ EPS 交换机（查表转发）→ 下一跳 NIC → GPU 显存。
3. Intra-host broadcast（NVSwitch）：gateway GPU 通过 NVSwitch 将同步后的梯度广播回所有 8 个 GPU。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- EPS 部署：Ethernet（RoCEv2, 100G-800G）或 InfiniBand（HDR/NDR, 200G-400G）。AI 集群常用 Clos/Fat-tree 拓扑（1:1 non-blocking 或 3:1 oversubscribed）。
- 关键组件：NIC（Mellanox ConnectX-6/7, 100G-400G）、交换机（NVIDIA Spectrum SN3700/SN5600, 32×400G 或 64×800G ports）、光模块/铜缆（QSFP28/QSFP-DD/OSFP transceiver 或 DAC cable）。
- EPS cost 构成（per 100G link）：transceiver $99 + NIC $659 + switch port $187 = ~$945/link。OCS cost（per 100G link）：transceiver $99 + NIC $659 + OCS port $520 = ~$1278/link。EPS 优势在于 switch port 单价更低，但需要 layer 多级交换机（Fat-tree 中 spine + leaf 两层），总 switch port 数量远多于 OCS。
- 在 MixNet 中，EPS 不是被替代而是被**增强**——OCS 作为额外的可重构高带宽域处理 EP 的动态流量，EPS 继续处理 DP/PP 的全局通信和 EP fallback。这种混合设计保持了对现有 EPS 基础设施的兼容性。

涉及论文标题：
- MixNet: A Runtime Reconfigurable Optical-Electrical Fabric for Distributed Mixture-of-Experts Training

## 存储分离（Storage Disaggregation）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 存储分离 = 把计算与存储从"一体化服务器（converged server）"中解耦，各自成为可独立扩容/缩容的资源池，经高速网络（如 NVMe-oF、RDMA）互访的数据中心架构范式。传统一体化服务器里存储盘绑定在具体服务器上，产生 stranded（搁浅）容量与扩展瓶颈；分离后计算节点与存储节点独立购买、独立升级，带来更高利用率、弹性与成本效率，已被主流云厂商与超大规模数据中心广泛采用（论文引 Socrates/Aurora 等云数据库与 Alibaba Luna-to-Solar 计算-存储网络）。
- 逻辑链：解耦 → 存储成为远端池（disaggregated storage pool）→ 计算节点通过 fabric 访问远端盘 → 网络成为关键性能路径 → 需要低开销高吞吐的远端存储协议（NVMe-oF/NVMe/TCP 或 RDMA）。论文语境：BoostX™-NTI 加速的正是分离存储的"发起端（initiator）"一侧——分离化把远端盘 I/O 的压力全部压到主机 CPU 的 NVMe/TCP 栈上，催生了 DPU 卸载需求。
- 训练系统语境的扩展（DisDP）：把「优化器状态/模型状态」这类训练数据也作为被分离的存储——MSDP（ZeRO-Infinity）把模型状态分片放 worker 本地 CPU/NVMe，DisDP 进一步把优化器整体搬到单台参数服务器（PS，配 12×SSD），worker 的 GPU/CPU 完全不再承载优化器状态存储；分离使最大可训模型不再受 worker 本地 CPU/SSD 容量限制（DisDP 恒定可训 1T 模型，ZeRO-Infinity 8 机仅 175B）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 分离存储系统的最小拓扑：计算节点（initiator，跑应用 + 存储协议栈）→ 数据中心交换机（论文用 Dell PowerSwitch Z9432F-ON，32×400GbE）→ 存储节点（target，跑 NVMe-oF target 软件，16× Samsung PM1743 SSD）。流程例子（云块存储读请求）：VM 内应用 read → virtio-blk → 主机 vHost/SPDK → NVMe/TCP initiator 栈把 NVMe Read 组装成 PDU → TCP 栈分段 → 100GbE 网络 → target 的 NVMe/TCP 栈解析 PDU → 本地 NVMe SSD 读数据 → 反向返回 data PDU 与 completion。
- 分离化的代价：协议处理（NVMe/TCP PDU 组装/解析 + TCP/IP 逐包处理 + 内存拷贝）全部落 CPU——论文实测 SPDK initiator 维持 200 Gbps 需 24 核、800 Gbps 网络预计 96 核；BlueField-3 SNAP 把栈搬到 DPU sidecore 也仅 9.6% line-rate。NTI 的解法是把该栈整体硬件化到 FPGA DPU，把分离存储的发起端开销从 CPU 和 sidecore 同时移除。
- DisDP 的流程例子（训练反向阶段）：GPU 产出 partial gradients → worker SmartNIC 发网 → SmartSwitch 网内聚合为单份梯度 → PS 收梯度到 CPU 内存 → 按 step-centric 流水从 SSD 读 12B 模型状态（Adam m/v + 主权重）→ CPU Adam → 写回 SSD，同时把 2B 参数副本推回各 worker。分离的代价是网络成为关键路径：必须靠 SmartSwitch 聚合把 PS 收发量降到与 worker 数无关，否则 16 worker 需 13~29 台额外 CPU 机器做 PS 才能线速（DisDP 图 5 模拟）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现层级自上而下：(1) 资源编排层——云平台把远端盘以块设备形式挂给 VM（论文的 SPDK vHost）；(2) 协议层——NVMe-oF 的 TCP/RDMA 两种 transport 二选一（TCP 复用商品以太网、TCO 低但 CPU 重；RDMA 需 RNIC + 无损网络调优，性能高但运维复杂、有 vendor lock-in 风险）；(3) 硬件层——通用 NIC + 主机 CPU，或 DPU/IPU 卸载（BlueField SNAP、Pensando、NTI）；(4) 可靠性层——Multipath/ANA 多路径与故障切换、keep-alive 超时检测。
- 使用要点：选择 transport 与卸载方案时权衡 TCO/部署复杂度/性能（论文结论：RDMA 适合极致低延迟，NTI 这类 FPGA 硬件卸载在商品以太网上同时拿到 line-rate 与 75W 低功耗）。信息缺口：论文未展开存储池端（target 侧）的资源调度与数据放置策略。
- DisDP 实现要点：单 PS（双 Xeon Gold 5320 + 12 SSD，26 GB/s 双向聚合 I/O）即可线速消费 100Gbps 聚合梯度并服务 100Gbps 参数（前向需 23.3 GB/s 内存带宽 + 11.6 GB/s SSD 带宽；反向需 99 GFLOPS + 349 GB/s 内存带宽 + 81.4 GB/s SSD 带宽）；SSD 用 PCIe Gen4 放 PS 侧，worker 侧无本地 SSD。信息缺口：论文未展开 SSD 内参数布局与预取调度。

涉及论文标题：
- BoostX™-NTI Fast, Scalable and Flexible Storage Architecture with NVMe-TCP Initiator Acceleration
- DisDP: Disaggregating Compute, Network, and Storage for Model-Sharded Data-Parallel Training

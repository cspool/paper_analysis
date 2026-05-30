## NVSwitch and NVLink (GPU互联架构)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NVSwitch 和 NVLink 是 NVIDIA 的 GPU 间高速互联技术，构成单节点内多 GPU 通信的基础架构。NVLink 是 GPU-to-GPU 的点对点直连链路（第三代 NVLink 带宽 300GB/s per link, A100 使用 12 条 NVLink 实现 600GB/s 双向带宽），NVSwitch 是一个独立的交换芯片，将节点内所有 GPU 的 NVLink 端口全互联，实现所有 GPU 对之间的全带宽 non-blocking 通信（A100 HGX 平台：6 个 NVSwitch 互联 8 个 A100 GPU，每 GPU 对外总带宽 900GB/s）。NVSwitch 是全双工、无阻塞的 crossbar 交换结构，任何 GPU pair 可同时以全带宽通信，不受其他 pair 影响。

从硬件架构角度拆解术语：
NVSwitch/NVLink 在 MoESys 的 Hierarchical AlltoAll 和 2D Prefetch 中的物理角色：
```
┌──────────────────────────────────────────────────┐
│                    Server Node                     │
│  ┌──────┐  ┌──────┐  ┌──────┐      ┌──────┐     │
│  │GPU 0 │  │GPU 1 │  │GPU 2 │ ...  │GPU 7 │     │
│  └──┬───┘  └──┬───┘  └──┬───┘      └──┬───┘     │
│     │NVLink  │NVLink  │NVLink       │NVLink      │
│     │300GB/s │300GB/s │300GB/s      │300GB/s     │
│     └────┬───┴────┬───┴────┬────────┴────┘      │
│          │        │        │                      │
│     ┌────┴────────┴────────┴────────────┐        │
│     │          NVSwitch (×6)             │        │
│     │    900GB/s per GPU, non-blocking   │        │
│     └────────────────────────────────────┘        │
│                                                     │
│  ┌──────┐  ┌──────┐          ┌──────┐             │
│  │NIC 0 │  │NIC 1 │  ...     │NIC 7 │             │
│  │100GbE│  │100GbE│          │100GbE│             │
│  └──┬───┘  └──┬───┘          └──┬───┘             │
│     │        │                  │                   │
└─────┼────────┼──────────────────┼───────────────────┘
      │        │                  │
   ┌──┴────────┴──────────────────┴───┐
   │     Leaf/Spine Ethernet Switch    │
   │    (跨节点通信，带宽低于NVSwitch)   │
   └───────────────────────────────────┘
```

NVSwitch 的带宽优势是 MoESys 2D Prefetch 的水平维度（NVLink）和 Hierarchical AlltoAll 阶段一（Intra-node）的核心物理基础。节点内通过 NVSwitch 做 AlltoAll 的成本（延迟 ~几微秒 + 900GB/s 带宽）远低于跨 NIC/switch（延迟 ~十几微秒 + 100Gbps 带宽）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- A100 HGX 平台：8 A100 GPU + 6 NVSwitch，每个 GPU 12 条第三代 NVLink（600GB/s total bidirectional），NVSwitch 全互联提供 900GB/s per GPU 的聚合带宽。
- H100 HGX 平台：8 H100 GPU + 4 NVSwitch，第四代 NVLink 每 link 450GB/s，18 条 link per GPU 提供 900GB/s total。
- B200 / GB200 NVL72 平台：使用第五代 NVLink + NVLink Switch，支持 72 GPU 的 NVLink domain，聚合带宽达 130TB/s。
- MoESys 的 Hierarchical AlltoAll 利用 NVSwitch 的拓扑特性——同节点 GPU 间数据传输不经过 NIC、不消耗网络带宽，仅占用 NVSwitch 容量（而 NVSwitch 容量远超实际使用需求）。跨 rank 的数据通过 NVSwitch 先搬运到同 rank GPU，再走 NIC 通信，实现了"免费"的数据重排。

涉及论文标题：
- MoESys: A Distributed and Efficient Mixture-of-Experts Training and Inference System for Internet Services

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Optical Circuit Switching (OCS，光电路交换) 是一种 Layer-1 交换技术，通过在交换机内部建立专用的光路（optical circuit/cross-connect）实现端口间的物理层直连，无需 packet buffer、header 解析或路由查表。数据以光信号形式在端到端电路上透明传输，交换节点仅控制光路通断（通过 MEMS 微镜、PLZT 波导、硅光子等技术偏转光束）。与 EPS（Electrical Packet Switching，逐包存储转发）相比，OCS 的核心优势是：端口带宽独立、无 buffer 拥塞、固定低延迟（仅传播延迟 + 交换配置延迟）、每比特功耗极低（无需 SerDes/Retimer）。代价是：重配置需要毫秒-分钟级机械/物理调整时间；端口间的连接拓扑必须预先配置（connection-oriented）；无原生多路复用（一条电路独占波长/光纤）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 MixNet 的硬件架构中，OCS 位于 scale-up（NVSwitch/NVLink）和 scale-out（EPS Fat-tree）网络的边界，形成**区域可重构高带宽域**。每台 GPU server 将 8 个 NIC 中的 2-6 个连接到 OCS（其余连接 EPS），OCS 根据 MoE 训练的实时 traffic demand 动态建立 GPU server 间的直连光路。

MixNet 中 OCS 的运转流程（以 Mixtral 8×22B，1024 GPU cluster 为例）：
1. **初始状态**：OCS 内部 cross-connect 为随机或前次训练的拓扑（所有端口处于已配置但非最优的连接状态）。
2. **Traffic 感知**：MixNet traffic monitor 收集各 EP group 的 all-to-all 通信需求矩阵 E[e_i][e_j]（预测或直接从 gate unit output 获取）。
3. **拓扑决策**：Greedy Algorithm 1 计算哪些 server 对需要专用 OCS 直连电路（优先 bottleneck pairs），生成 NIC 级别的 TX/RX 端口映射 S。
4. **物理重配置**：Topology controller 通过 TL1 commands over Ethernet 向 OCS 发送 S。OCS 内部 MEMS 微镜阵列偏转光束，重新建立 TX_i → RX_j 的物理光路连接。Polatis 576×576 OCS 的重配置时间：平均 41-47ms（1-16 pairs），99th percentile <70ms。
5. **数据传输**：光路建立后，两端 GPU 通过 RDMA over RoCEv2 在专用光路上直接传输 EP all-to-all 数据（无交换机 buffer、无排队延迟、无 packet loss）。
6. **动态调整**：每个 MoE layer 最多重配置 2 次（FP 一次 + BP 一次），在 expert computation 期间隐藏重配置延迟。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
OCS 的实现技术和部署：
- **MEMS-based OCS**（Google Palomar 136×136, Calient 320×320, Polatis 576×576）：利用 MEMS 微镜阵列反射/偏转光束，重配置 10-25ms，端口成本 ~$520/port。Google 已将 MEMS OCS 部署于 TPU v4 supercomputer（Lightwave Fabrics）。
- **Robotic OCS**（Telescent 1008×1008）：使用机器人物理插拔光纤 patch panel，重配置数分钟，端口成本 ~$100/port。TopoOpt 采用此方案。
- **Silicon Photonics**（Lightmatter 32×32, iPronics）：利用片上 Mach-Zehnder 干涉仪或微环谐振器调谐光路，重配置 7μs-μs 级，端口数有限（≤32）。SiP-ML 和 Lightmatter Passage 探索此类方案。
- **PLZT-based OCS**（EpiPhotonics 16×16）：电光效应超快切换（10ns），端口数极少。
- 部署模式：(a) 网络核心 OCS（Google Jupiter Evolving, Lightwave Fabrics）——OCS 连接 EPS 交换机集群；(b) 边缘 OCS（MixNet 的区域可重构域）——OCS 直接连接 GPU server 的 NIC，位于 scale-up/scale-out 边界；(c) Chip-level OCS（Lightmatter Passage）——co-packaged optics 直接连接到 GPU die。
- 关键 trade-off：端口数 vs 重配置速度（Table 2）——大端口 → 慢重配置（机械/机器人，分钟级）；快重配置 → 小端口（硅光子/PLZT，μs-ns 级）。MixNet 的区域可重构设计利用 MoE 通信的局部性（EP group ≤128 GPU）绕过了此 trade-off。
- 工程挑战：(a) 光模块/NIC 需支持快速 CDR 锁定和信号恢复（burst-mode transceiver, fast-locking CDR）；(b) 重配置期间链路中断（需在计算阶段隐藏延迟）；(c) OCS 无 packet buffer，不能像 EPS 那样 buffer 拥塞流量。

涉及论文标题：
- MixNet: A Runtime Reconfigurable Optical-Electrical Fabric for Distributed Mixture-of-Experts Training

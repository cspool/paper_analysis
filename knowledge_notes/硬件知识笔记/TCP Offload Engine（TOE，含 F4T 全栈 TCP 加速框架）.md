## TCP Offload Engine（TOE，含 F4T 全栈 TCP 加速框架）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- TOE = 把 TCP/IP 协议处理（分段/重组、校验和、重传、拥塞控制、连接状态管理）从主机 CPU 下放到专用硬件（ASIC NIC、DPU、FPGA IP）的引擎，主机只发"发送字节区间"级请求。传统 TOE（如 Chelsio 网卡）是固定 ASIC；F4T（ISCA 2023，SNU HPCS Lab）提出 FPGA 全栈 TCP 加速框架——解决 FPGA TOE 的两大瓶颈：有状态操作的 stall 与多存储模块间的 TCP 状态管理，并提供免改应用的软硬件全栈；100Gbps 打满仅需 2 核，比 Linux TCP 栈省 64% CPU 周期（Web 证据：ACM DL 10.1145/3579371.3589090、SNU HPCS 实验室页）。
- NTI 与 F4T 关系：NTI 的 TOE 基于 F4T 框架构建（论文 [47]），选择 FPGA/C/HLS TOE 的理由是复杂 TCP 算法与鲁棒性/安全修复可快速更新；标准 TCP 机制（Cubic、SACK）保证大规模连接下的工程成熟度（§VI-A4 多目标压力测试依赖这点）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- NTI 中 TOE 的职责与交互（Fig.6）：TX 方向——PDU Header Generator 通知 TOE"有 PDU 待发"，TOE 维护每连接状态（如最老未确认序列号）决定何时发出 TX 读请求，向 PDU Stitcher 的 TX virtual buffer 按字节区间取 payload（触发按需 DMA），再组 TCP 包发出；RX 方向——TOE 收包校验/重组/应答（Cubic + SACK），把有序 TCP payload 交给 PDU Parser。
- 资源账本：TOE 占 225K LUT/332K FF/264 BRAM/132 URAM——三 IP 中 LUT 最多（TCP 状态机与连接状态存储重）；对连接数扩展，论文声明复用 F4T 验证过的通信层（§VI-A4：1-to-N 多 target 压测持续饱和线速，连接交错/乱序/PDU 错位无退化）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 通用实现：连接状态表放片上/DRAM（F4T 管理多存储模块的 TCP 状态）、发送引擎按字节区间回读（支持重传任意字节范围）、接收引擎做顺序化重组。与 TOE 相邻的体系问题：TOE 消费的是字节流，而上层 NVMe/TCP 消费的是 PDU——二者的边界错位正是 NTI Virtual buffer 要解决的核心（见 Virtual Buffer 条目）。
- 使用场景：高带宽存储/网络 DPU 的传输层；要求 TOE 具备"重传任意历史字节区间"能力（TCP 语义），与 PDU 生成侧的解耦设计要配套。信息缺口：论文未说明 NTI TOE 相对 F4T 原版的具体改动与连接数上限。

涉及论文标题：
- BoostX™-NTI Fast, Scalable and Flexible Storage Architecture with NVMe-TCP Initiator Acceleration

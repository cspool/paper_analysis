## NVMe-oF 与 NVMe/TCP（含 PDU 与 PDU 解析）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- NVMe-oF（NVMe over Fabrics）= 把 NVMe 命令/完成/数据从本机 PCIe 扩展到网络 fabric 的协议族（RDMA、FC、TCP 等 transport 可选）。NVMe/TCP = 其中的 TCP transport：用 PDU（Protocol Data Unit）封装 NVMe 命令、完成与数据，经标准 TCP/IP 传输——因复用商品以太网、可跨无损/有损异构网络可靠运行而成为生产部署主流。
- PDU 是消息级封装单位，含 PDU header（含 PDU type/opcode、长度等）+ 可选 payload。标准 PDU 类型（Web 证据：NVM Express TCP Transport Spec Rev 1.1，ratified 2024.08）：ICReq/ICResp（连接初始化）、H2CTermReq/C2HTermReq（连接终止）、CapsuleCmd/CapsuleResp（命令/响应 capsule，封装 SQE/CQE）、H2CData/C2HData（写/读数据）、R2T（Ready to Transfer）。方向错误（H2C vs C2H）为致命传输错误。
- 关键困难（论文 §II-A2）：TCP 是连续字节流、NVMe/TCP 是离散 PDU，二者粒度不同——一个 TCP 包可装多个 PDU，一个 PDU 也可能跨多个 TCP 包。接收端必须做 PDU parsing：顺序扫描 TCP payload，读每个 PDU 头定长，再按长度定位并切出下一个 PDU。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 发起端（initiator，主机侧）流程（一次读）：主机 NVMe Read（带 PRP 数据缓冲地址）→ initiator 栈生成 CapsuleCmd PDU 头 → 经 TCP 发送 → target 解析后回 C2HData PDU（读数据可能跨多个 PDU）+ CapsuleResp PDU（完成）→ initiator 栈对每个收到的 TCP payload 做 PDU parsing 切出 PDU 头与数据 → 数据落到 PRP 指定缓冲 → 从 CapsuleResp 提取 NVMe completion 上报。注意 NVMe 与 NVMe/TCP 的地址空间鸿沟：PRP 指向发起端主机内存，远端 target 看不到，故 target 只回"消息"、由发起端自己按记录好的 PRP 放数据。
- 系统级比较（论文 §II-A3）：NVMe/TCP vs NVMe/RDMA——TCP 低 TCO、可部署性强（标准以太网、无需 PFC/ECN 调优），但 CPU 开销高；RDMA 由 RNIC 卸载 transport、性能高 CPU 低，但需昂贵 RNIC + 无损网络且对丢包敏感、机制未完全标准化存在 vendor lock-in。NTI 的目标是"保留 TCP 的运维好处、拿到 RDMA 级性能"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 软件实现：Linux 内核 nvme-tcp host 驱动（红帽文档配置）、SPDK 用户态 NVMe-oF target/initiator（polling、zero-copy、kernel bypass）；DPU 实现：NVIDIA BlueField SNAP（sidecore 软件栈）、NTI（FPGA 硬件把 PDU 生成/解析做成固定流水）。队列模型：标准 NVMe/TCP 中 NVMe 队列与 TCP 连接 1:1（Web 证据：专利与规范），NTI 的 Command Scheduler 解耦了这一映射（NVMe 命令可路由到任意 NVMe/TCP 队列）。
- 使用要点：PDU parsing 是 CPU 大头（顺序扫描 + 长度跳转），也是 NTI Virtual buffer 硬件化的核心对象（边收边切、按 PRP 直写主机内存、不落中间缓冲）。信息缺口：论文未给出 NTI 支持的 PDU 类型全集（仅明确 C2HTerm、数据 PDU、完成 PDU 等错误场景）。

涉及论文标题：
- BoostX™-NTI Fast, Scalable and Flexible Storage Architecture with NVMe-TCP Initiator Acceleration

## RDMA（远程直接内存访问）与 RNIC（含 QP/CQ、one-sided/two-sided verb）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- RDMA 是一种让一台机器直接读写另一台机器内存、全程绕过远端 CPU 与操作系统的网络技术，由 RDMA 网卡（RNIC）硬件完成网络栈卸载：靠内核旁路（kernel bypass，用户态直接向网卡发请求）、零拷贝（数据不经过协议栈缓冲）与 CPU 卸载三大机制，达到 ~2 µs 级往返延迟与 100/400 Gbps 带宽（Web 证据：Intel RDMA 文档与 Continuum Labs 教程）。通信由 QP（Queue Pair，含 SQ 发送队列与 RQ 接收队列）承载，完成事件由 RNIC 写入 CQ（Completion Queue）形成 CQE，应用轮询或事件通知。verb 分两类：one-sided（READ/WRITE/CAS/FAA，远端 CPU 完全不参与，直接操作注册内存区）与 two-sided（SEND/RECV，需双端 CPU 配合、消息传递语义）；内存须先注册（MR，生成 lkey/rkey 访问凭证）。论文语境：RDMA CAS/FAA 是分布式锁、无锁索引（RACE）、事务协调（DrTM）的基石。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 一个 one-sided 原子请求的硬件流程：应用把 WQE 写入 SQ → 敲 doorbell（MMIO）→ RNIC 经 DMA 取 WQE、翻译虚拟地址 → 服务端 RNIC 对注册内存执行 CAS（经内部锁定表槽 + PCIe RMW，见下一条）→ 完成时 RNIC 把 CQE 写入 CQ → 客户端 poll CQ 得知结果。全程服务端 CPU 不参与。论文中 Fusa-Driver 在 ibv_post_send 路径上拦截该流程、按策略改写目的地（RNIC 或 Fusa-Server 缓冲），所以 QP/CQ 与 doorbell 机制是整个调度的硬件支点。
- 关键硬件配套：大页（论文用 2 MB）减少 RNIC 页表翻译开销；DDIO 让 RNIC 写直达 LLC（见 DDIO 条目）；PCIe 带宽/请求率决定小请求瓶颈（Web 证据：Collie 论文指出小消息受 WQE/MMIO doorbell 请求率限制）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 三种承载：InfiniBand（原生 RDMA、无损，论文即用 IB）、RoCEv2（以太网 UDP）、iWARP（TCP）。用户态用 libibverbs（rdma-core），数据路径 ibv_post_send/ibv_poll_cq 无系统调用（写 WQE + MMIO doorbell，见"libibverbs"条目）。典型使用：FaRM/HERD 类 KV 系统、RACE/Sherman 索引、DrTM/FaSST 事务。论文测试台：100 Gbps ConnectX-6 InfiniBand + SN2700 交换机 + MLNX OFED v24.10-2.1.8。

MTIA 300 补充视角（ISCA'26，内置 RDMA NIC 的芯片内化）：MTIA 300 把 12 个 800 Gbps RoCE RDMA NIC 直接封装进 2 个网络 chiplet，RDMA 数据路径完全在包内完成（不经 PCIe、主机不参与数据面）：HCCL 以 RDMA verbs 建 QP（ibv_create_qp/ibv_modify_qp 至 ready 态、ibv_get_async_event 捕获非 WC 错误），数据路径用 MTIA streaming interface 提交、经 NIC interface 单 FIFO 分发到 12 NIC 的 express doorbell（WR 即 doorbell 写，省 800 ns HBM ring 读）；每 NIC 1024 QP（24576 outstanding WR）、因去 QP caching 限制 1100 active QP，12 NIC 共 13056 QP 可切分/共享于 scale-up/scale-out 域。与通用 RNIC（如 ConnectX，见本条目）差异：MTIA NIC 去掉 virtual switching/TC offload/QP cache 等通用功能、加 AXI steering tag 分流，凸显"训练加速器专用 RDMA NIC"的定制方向。

R2D2 补充视角（ISCA'26，disaggregation 网络传输）：DDC 的 disaggregation traffic（sub-10µs、40-400 Gbps）标准承载即 RDMA-capable 网络——专用、全连接、RDMA fat-tree（Infiniswap 用 one-sided RDMA 绕过远端 CPU、FastSwap、LegoOS 的 RDMA 分布式 OS）。在 R2D2 数据面上：compute 节点发起 RDMA_READ → RNIC（Broadcom P1400GD 400Gb）DMA 组装请求 → 400 GbE 链路经机器人建立的被动光纤直连 → 目标 RNIC DMA 写入并返回，无交换机逐包转发/共享瓶颈，全程单跳满带宽。RNIC 延迟 ~500ns 占主导；R2D2 加 ~10m 光纤（<50ns 传播）+0.1-0.3dB 插损（低于 400GBASE-SR8 的 1.9dB 预算），相比 fat-tree 省去 ToR 455ns 交换延迟 → 端到端省 ~42%。R2D2 对 host 透明、兼容 RDMA——其价值在网络本身（互连+runtime），与 RDMA/内存节点侧技术正交。评估 NIC：400G 用 P1400GD、100G 用 Mellanox ConnectX-5。

涉及论文标题：
- Breaking Barriers in Atomic Scaling: A Hardware–Software-Collaborated Framework to Deconstruct RDMA Atomic
- MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines
- R2D2 Robotized Reconfigurable Network for Disaggregated Datacenters

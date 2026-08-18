## Intel DSA / IAA（Data Streaming Accelerator / In-memory Analytics Accelerator）

术语解释
Intel Sapphire Rapids 起的核外加速器：DSA 做内存数据搬移/CRC 等"数据中心税"操作，IAA 做解压/压缩/扫描/CRC；二者都是论文 OCA 抽象的商业代表。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DSA/IAA 是片上、挂 PCIe/片上总线上的设备型加速器：软件提交 64B 工作描述符（源/目标地址、长度、操作码、完成记录地址），硬件异步执行并写完成记录，软件轮询完成。提交经 MMIO portal（每 portal 独占 4KB 页，可经页表隔离映射到用户态）：`MOVDIR64B` posted 写（快、无反馈）或 `ENQCMD/ENQCMDS`（非 posted、返回成功/retry）。work queue 分 dedicated（配 MOVDIR64B）与 shared（配 ENQCMD，支持 PASID 虚拟化）。Linux `idxd` 驱动 + Intel DML/QPL 库是标准软件栈。论文对 OCA 的批判直接以这类设备为原型：MMIO 调用非推测（须到 ROB 头）、任务启动串行、调用与轮询之间要 fence，小任务下开销致命。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
论文的 baseline 使用方式：LLC-attached OCA 配置模拟"最常见 OCA"——挂 LLC、用 load/store 控制（等价 MMIO 描述符语义）、但换装 UTE 的流式 out-of-core 访存接口（继承 stream 支持），从而把对比隔离在调用模型上；L2-attached OCA 则用 RoCC-like 指令（另一 OCA 控制方式）。结果：ATX NCA 较 LLC OCA 快 2.6–9.4×（任务越小差距越大）；OCA 任务足够大时性能趋近 NCA，代价是丢失细粒度交错与更大的片上 scratchpad。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DSA 操作类型：内存 move、fill、compare、CRC32C、DIF；IAA 操作类型：deflate 解压/压缩、CRC、scan。典型软件流程：申请 portal → 构造描述符（DML/QPL 封装）→ ENQCMD 提交 → 轮询完成记录 status 位（或 MSI 中断）→ 消费结果。适合大批量、异步、粗粒度数据操作（与核计算解耦、容忍毫秒级延迟）；不适合小任务频繁调用（描述符提交 + 轮询开销 > 计算本身）。这正是 ATX 论文想用 NCA 填补的中间地带。

DarkStream（ISCA'26）补充：DSA 用 group 抽象隔离客户端（每组一个或多个 WQ + engine + 组内 arbiter），但隔离边界之下的 I/O fabric interface 吞吐仍设备级共享。DarkStream 在 Xeon Platinum 8558 实测：单 DSA 设备 128 个 WQ 项、4 个 engine、solo-run 峰值约 30 GB/s；co-run(同 DSA) 吞吐在约 15 GB/s 饱和且延迟同步分化，co-run(独立 DSA) 不饱和——证明残余争用来自设备内数据搬移而非 descriptor 处理。据此构建隐蔽信道（slot 调制 1-byte Memory Move，峰值 129 Kbps @147 KHz）与网站/DL 模型指纹侧信道（97.03%/99.17%）。侧信道受害端用 Intel DTO 库把 libc memcpy/memset 卸载到 DSA（>8 KB 阈值）：memset→Fill、memcpy/memmove→Memory Move。缓解讨论：DSA 独占分配、engine 时间复用（吞吐 30→7.5 GB/s）、性能计数器检测。

涉及论文标题：
- ATX: Accelerator Task Extensions
- DarkStream: Exploiting Internal Throughput Contention in Data Streaming Accelerator for Timing Attacks

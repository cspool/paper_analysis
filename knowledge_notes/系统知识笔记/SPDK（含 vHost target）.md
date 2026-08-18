## SPDK（含 vHost target）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- SPDK（Storage Performance Development Kit）= Intel 发起的开源用户态存储加速套件（spdk.io）：绕过内核块层与文件系统，以用户态驱动 + 轮询模式 + 无锁队列 + 大页/零拷贝访问 NVMe 设备，并提供 NVMe-oF target/initiator、vhost、bdev 等模块。论文中三个角色：(1) 软件基线——SPDK NVMe/TCP initiator 维持 200 Gbps 需 24 核（FIO 4KiB 随机、32 线程、QD256，引自 Intel SPDK perf report 24.05）；(2) target 端软件——微基准与 MLPerf Storage 实验里 target 跑 SPDK NVMe/TCP；(3) 云场景——SPDK vHost 做存储虚拟化给 QEMU/KVM 虚拟机挂 virtio-blk。
- 逻辑链：内核存储栈（bio→request queue→驱动→中断）在高 IOPS 下软中断/上下文切换/锁开销大 → SPDK 把整个 I/O 栈搬到用户态轮询线程（reactor）→ 裸金属级 IOPS 但吃满绑定的核 → 催生"SPDK 之后的下一棒"：把存储协议栈继续卸载到 DPU 硬件。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 论文云场景流程（NTI + SPDK vHost）：每个 vHost 分配 1 核 + 8GB 内存，QEMU VM 的 virtio-blk 请求 → vHost 用户态处理 → 经 bdev 层下发给 NVMe/TCP initiator 路径 →（软件基线：SPDK initiator 栈抢占 vHost 的核，1 核时严重核争抢；NTI：initiator 栈整体在 DPU 硬件，vHost 独占核）→ 网络 → SPDK target。结果：NTI 使聚合带宽在 1/4/8/16 vHost 核下达 2.64×/2.41×/1.71×/1.10×，即同样带宽 SLA 下可承载约 2× VM 密度、基础设施减半。
- 该例子说明系统层权衡：SPDK 类用户态栈性能极高但"性能=CPU 核数"，核数与 VM 数互相挤占；把 NVMe/TCP 栈移出主机后，vHost 核全部用于 VM 服务，SLA 容量翻倍。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现要点（Web 证据：spdk.io 文档）：reactor 模型把每个核绑定一个 poller 循环；vhost target（vhost_tgt 应用，配 -S 参数）为 QEMU 提供 virtio-blk/virtio-scsi 设备；NVMe-oF target（nvmf_tgt）支持 TCP/RDMA transport。论文 target 配置为 SPDK NVMe/TCP + 16× Samsung PM1743 SSD。
- 使用场景：作为高性能基线（CPU 效率对比）、target 软件、VM 存储虚拟化层，以及与 DPU 硬件栈协作（MangoBoost 的 NTI 控制面用 SPDK RPC 配置、nqn 形如 nqn.2019-07.io.spdk:cnode0，Web 证据：Mango SDK 文档）。信息缺口：论文未给出其 SPDK target/initiator 的版本号与调优参数。

涉及论文标题：
- BoostX™-NTI Fast, Scalable and Flexible Storage Architecture with NVMe-TCP Initiator Acceleration

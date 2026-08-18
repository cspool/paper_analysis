## NVMe Multipath 与 ANA（Asymmetric Namespace Access）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- NVMe Multipath = NVMe 标准的多路径机制：一个 namespace 可经多条路径（多个控制器/端口）访问，主机 NVMe 驱动把同一 namespace 的多个控制器聚合成一个块设备，提供可用性与带宽聚合。配置模式：active/active（所有活跃路径并发使用）与 active/passive（备用路径 standby，主路径故障后切换）。ANA（Asymmetric Namespace Access，NVMe Base Spec §5.14.1.12）= 每条路径的 ANA state（optimized / non-optimized / inaccessible 等），由 target 上报给主机，指导主机选择路径。
- 逻辑链：网络拓扑变化/存储控制器迁移是分离存储的日常运维事件 → 主机必须感知哪条路径可用/最优 → ANA state 提供这一感知（Web 证据：SPDK 仅使用 ANA optimized 路径，除非没有 optimized 路径）→ 故障时主机把 I/O 切到存活路径。论文用它验证 NTI 的"operational resilience"设计目标。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 论文实验（initiator + target 经 2×100GbE 双链路）：(1) active/active——带宽初始 200 Gbps，一条路径故障后该路径 I/O 丢失，NTI 检测到故障把未完成命令经存活路径重传（100 Gbps），重连后恢复 200 Gbps；(2) active/passive——初始 100 Gbps，主路径故障后备用路径立即顶上；(3) ANA——行为类似 active-active，但 ANA state 迁移在微秒级被检测到，存活路径无停顿地承载全部流量。三种场景下主机视角 I/O 不断流，无需重建 I/O session。
- 系统含义：多路径/ANA 把"硬件故障/拓扑变更"从服务中断降级为带宽瞬时波动；硬件卸载方案（NTI）必须与这些软件栈机制（内核 nvme multipath、SPDK multipath）无缝配合，这正是 NTI 以标准 NVMe 接口 + 协议合规换取的能力。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现（Web 证据：SPDK 文档 spdk.io/doc/nvme_multipath.html 与 nvmf_multipath_howto.html）：initiator 侧 bdev NVMe 模块支持 failover 与 multipath 两模式；multipath 模式支持 ANA，active-passive 用"缓存的首个 optimal 路径"（bdev_nvme_set_preferred_path 可指定偏好路径与 failback），active-active 用 round-robin 或最小队列深度选路；target 侧创建 subsystem 时加 -r 开 ANA reporting，用 nvmf_subsystem_listener_set_ana_state 改路径 ANA state。
- 使用场景：分离存储的高可用设计、滚动升级/节点替换时的路径切换、多路径带宽聚合；也是评估 DPU 硬件"错误时行为"的测试面（论文 §VI-A3 即用 FIO + 拓扑变更验证 NTI）。信息缺口：论文未说明 NTI 实验里 ANA 状态机由主机驱动还是 target 驱动，仅报告"微秒级检测"。

涉及论文标题：
- BoostX™-NTI Fast, Scalable and Flexible Storage Architecture with NVMe-TCP Initiator Acceleration

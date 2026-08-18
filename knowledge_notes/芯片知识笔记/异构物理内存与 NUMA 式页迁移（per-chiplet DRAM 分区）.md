## 异构物理内存与 NUMA 式页迁移（per-chiplet DRAM 分区）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PhaseWeave 的内存子系统：所有 chiplet 共享统一物理地址空间、由单一 OS 管理，但物理上异构——DRAM 按 memory channel/DIMM 粒度被组织成每个 chiplet 专属的分区，各分区的延迟与带宽按该 chiplet 服务的 phase 类别配置（表 III：fast-memory chiplet 25.60GB/s、22 cycles 延迟 vs 其他 17.06GB/s、15 cycles），软件看到平坦地址空间而底层是 non-uniform、heterogeneous 的内存基板。配合 NUMA 式页迁移：新分配内存优先落在执行请求的 chiplet 本地分区（初始局部性）；随访问模式漂移，周期性评估访问计数并把热页提升/迁移到主导访问的 chiplet 分区。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
在芯片物理组织上，这是"每 chiplet 直连自己的 DRAM 通道/DIMM"的物理内存分置（对比 monolithic 服务器的均匀内存带宽/延迟）。运转流程：GEMM phase 线程在 compute chiplet 执行，其分配落在 compute chiplet 本地分区（Mem BW 17.06GB/s）；当线程迁到 fast-memory chiplet 跑 DeepCopy phase，其热页经页迁移策略（周期读访问计数、promote hot pages，论文明确"NUMA-like"）逐步搬到 fast-memory 分区，从而按 phase 需求匹配带宽/延迟。与 MCM-GPU 的 NUMA 效应同源：inter-chiplet 访存贵于 intra-chiplet（vault 中 L1.5 Cache in Multi-Chiplet GPU 笔记），故"页跟随执行"是把异构带宽/延迟变成收益而非惩罚的关键。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：内存控制器按 chiplet 分区独立建模（DRAMSim3 每分区参数化带宽/延迟），页分配器做 first-touch 本地分配，OS 侧用类似 NUMA page migration 的机制（周期性 access counter 采样 + 热页迁移，类似 Linux auto NUMA balancing 的思路）适配动态 phase 漂移。用途：让异构内存资源（带宽向 fast-memory chiplet 倾斜）可被运行时按 phase 利用，是"硬件异构"在内存维度上的落地；对后续研究，它展示了"统一地址空间 + 物理异构分区 + OS 级页迁移"三件套，区别于 CXL 内存扩展（CXL 是经协议外扩、这里是封装内每 chiplet 直连分区）。

涉及论文标题：
- PhaseWeave Phase-Aware Execution on Heterogeneous Chiplet Architectures for Datacenters

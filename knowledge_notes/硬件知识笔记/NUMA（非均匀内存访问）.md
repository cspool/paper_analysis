## NUMA（非均匀内存访问）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NUMA（Non-Uniform Memory Access）是多 socket 服务器中内存访问延迟/带宽随"访问哪个 socket 的内存"而异的架构：本地内存快、跨 socket 远程访问需经 socket 间互连跳转（inter-socket hop），增加访问距离、降低带宽。SHD 只挂一个 socket 时，其他 socket 访问它必须走 inter-socket 跳。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
本论文四 socket NUMA 实验（YCSB-B）：1N1S_local（SHD 挂单 socket）其他 socket 的 CXL 访问经 inter-socket 跳，延迟最高 1.7× 于单 socket、吞吐受 inter-socket 带宽限制；4N1M_private（每 socket 一个 MHD head）延迟降 28%，但访问他人 head 仍要 inter-socket 通信；4N4S_SWopt（PBR 交换机把各 socket 内存路径并入共享 CXL 池，多端口接口直连所有 NUMA 节点）跨 socket 延迟近乎一致、吞吐 4×——硬件确定性 fabric 从根上消除 NUMA 依赖（inter-socket 通信被 fabric 内的统一内存路径替代）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
x86 多 socket 服务器每个 socket 一个 NUMA node，OS 按 NUMA 距离做页放置/迁移与调度（Linux autonuma；TPP、NeoMem 等 CXL tiering 系统在 NUMA/CXL 间做分层放置）。本论文表明 CXL 交换机可把 socket 级内存路径统一为共享池，消除 inter-socket 依赖，是 fabric 级 NUMA 无关化的硬件方案。

Vistara 补充视角（ISCA'26，CXL 内存作为 CPU-less NUMA 节点）：Vistara 把 CXL-attached DDR4 暴露为独立"无 CPU 的 NUMA 节点"，与本地 DRAM 节点分离，让内核内存管理独立识别/管理各 tier（分配、放置、访问策略按 tier 时延/带宽特性定制）。关键做法：① Linux CXL 驱动（非 BIOS 配置）把 CXL 内存 online 为 ZONE_MOVABLE，确保内核非可迁移分配（页表、slab 等）不落 CXL，保迁移性与可靠性；② ACPI CEDT（CXL Early Discovery Table）描述 CXL 内存配置、HMAT（Heterogeneous Memory Attribute Table）暴露各内存 NUMA 节点的相对时延/带宽，供内核分层决策；③ 默认内存策略：优先本地 NUMA 分配、本地耗尽才溢到 CXL，配合 NUMA demotion（numa_demotion_enabled=1）与 NUMA balancing（numa_balancing=2）做跨 tier 迁移；④ cgroup cpuset.mems 控制器可把某工作负载的分配限制在本地 NUMA 节点（CXL opt-out），无需 BIOS/重启即可切换。生产验证：本地:CXL 3:1 时 Web/DPP 无回归（<1% 偏差、CXL spill ≈10%），2:1 仍无回归，1:1 才出现可测回归（Web MIPS -3%/IPC -5%、DPP MIPS -12%）。

涉及论文标题：
- A Silicon-Proven Unified Low-Latency CXL Controller and Port-Based Routing Switch for Memory-Centric Fabrics
- Coarse-Grained Duplication First, Fine-Grained Deduplication Later: Duplication-Centric Multi-GPU Memory Management（UVM 多 GPU 中的非一致性 NUMA：无 cache 一致性，远端访问不可缓存、暴露完整远端延迟，是页迁移/复制与 CDFD 粗粒度复制去重的核心动机）
- Vistara: Making CXL Real—Full Path from ASIC Design and OS Support to Hyperscale Deployment

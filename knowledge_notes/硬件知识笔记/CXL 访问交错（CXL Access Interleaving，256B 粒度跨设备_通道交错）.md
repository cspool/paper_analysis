## CXL 访问交错（CXL Access Interleaving，256B 粒度跨设备/通道交错）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CXL 访问交错指把主机对 CXL 内存的访问在多个 CXL 设备/通道/rank/bank 之间均衡分布的硬件映射机制，让并发请求同时命中不同链路与存储组件，最大化聚合带宽并避免单链路热点。Vistara MemServer 采用两级交错：① 设备级——CPU host-bridge 以 256B 粒度把连续内存块交替映射到 2 个 Vistara ASIC（各 PCIe Gen5 x8），两设备可同时服务访问；② 设备内——每个 Vistara 内再跨 DIMM、rank、bank 交错，把流量均匀铺到所有通道与组件。生产实测 CXL 峰值 ≈76 GB/s（2 设备聚合）、单设备 48 GBps（ALL-Read 62% 理论峰值）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程（一次 64B cacheline 读 + 相邻地址）：地址 0x0 落在 Vistara 0（256B 粒度取模）→ Vistara 0 的 DDR 控制器按地址位把访问映射到 DIMM0/rank0/bank X；地址 0x100 落在 Vistara 1 → DIMM1/rank0/bank Y——两条请求并发走两条 PCIe Gen5 x8 链路与两套 DDR4 通道，无争用。作用：① 均衡 CXL link 负载、避免单设备/单链路成为瓶颈；② 允许同时访问两设备提升聚合带宽；③ 设备内 bank 级并行降低排队时延；④ 对容量受限负载，CXL 带宽利用率仅 <10%（生产本地:CXL 带宽比 ≈10:1），交错保证稀疏的冷页访问也均匀分布，避免热点放大时延。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：主机 host-bridge 内的地址哈希/取模逻辑（无需软件参与），交错粒度由平台固件/ACPI 配置；Vistara ASIC 内部 DDR 控制器按地址位做 channel/rank/bank 映射（标准 DDR 控制器地址重映射）。使用场景：任何多 CXL 设备扩展系统（多扩展器、多通道）都需要交错来分摊流量；与本地 DDR 的 channel interleaving、HBM 的 bank 交织同一原理。论文未提供开源实现；该设计是平台集成特性，可在 PCIe 拓扑内（MCTP/OSPM ACPI 表）配置。对后续系统的启示：交错粒度（256B）需与 cacheline（64B）和 CXL.mem 请求粒度匹配，过粗会偏斜热点、过细会增加命令/重放开销。

涉及论文标题：
- Vistara: Making CXL Real—Full Path from ASIC Design and OS Support to Hyperscale Deployment

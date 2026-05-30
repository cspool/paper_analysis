## Monolithic 3D-Stackable DRAM (Mono3D DRAM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Monolithic 3D-Stackable DRAM（Mono3D DRAM）是一种新型三维堆叠 DRAM 技术，通过在单一 wafer 上顺序制造（monolithic fabrication）多层水平 1T1C DRAM cells，取代传统 HBM 的 TSV die-stacking 工艺。与 HBM 将独立制造的 DRAM dies 通过 TSV 垂直堆叠不同，Mono3D DRAM 直接在 wafer 上逐层沉积和刻蚀构建，借鉴 3D NAND Flash 的制造工艺（layer-by-layer deposition, high-aspect-ratio etching, dense vertical integration）。其核心特征是：垂直 bitline（BL）连接 + wordline（WL）staircase 结构，加上 Cu-Cu hybrid bonding（1μm pitch）连接存储层和逻辑层。Mono3D DRAM 的关键优势：(1) 密度——Stratum 论文中实现 2.156 Gb/mm²（5.2× DDR5），1024 层，单芯片 32 GB；(2) 内部带宽——19.01-34.34 TB/s（远超 HBM ~800 GB/s per stack）；(3) 成本——避免 TSV 制造和 die bonding 的高成本/低 yield；(4) 散热——thinner dies + 更好的垂直热传导。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 Stratum 芯片设计中，Mono3D DRAM 的组织如下：
- **Bank 结构**：1024 BLs × 1024 WLs = 1 MAT，1024 MATs = 1 bank，bank capacity = 1 Gb，bank area = 0.439 mm²（35nm feature size，BL pitch 70nm，WL pitch 1μm）。
- **WL Staircase**：垂直堆叠的 WL 通过 staircase 结构路由，底层 WL 因 staircase 延伸导致寄生 RC 增大，访问延迟线性增加。Stratum 基于这一特性定义 8 个 timing tiers（tRCD 从 2.29ns 到 22.88ns）。
- **Hybrid Bonding 接口**：DRAM die 通过 Cu-Cu hybrid bonding（1μm pitch）连接到 7nm logic die，利用 BEOL metal routing 实现高密度垂直互联。对比 HBM 的 TSV（10μm pitch），Mono3D DRAM 的垂直互联密度提升约 5×。
- **CUA 结构**：高电压（HV）DRAM 外围电路（D/Q buffer, level shifter, address decoder）在 32nm CMOS-Under-Array 工艺中实现，低电压（LV）逻辑在独立的 7nm logic die 上通过 hybrid bonding 连接。

数据访问流程：xPU → silicon interposer → logic die PHY → ring network → PU → PE → local memory controller → hybrid bonding → Mono3D DRAM bank。xPU-DRAM 间通过 1024-bit I/O @ 6.4 Gbps/pin（与 HBM3 相同标准）通信，但大部分 NMP 计算中的数据在 Mono3D DRAM 内部以 19-34 TB/s 带宽闭环，仅最终的 token I/O 穿越 interposer。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Mono3D DRAM 目前仍处于学术研究和早期工业探索阶段。制造方面，Samsung 于 2023 年展示了 3D DRAM 的可行性（VLSI 2023），采用垂直堆叠的 1T1C 结构和 staircase WL 路由。Stratum 论文使用 Coventor SEMulator3D 工艺模拟器 + NeuroSim 电路仿真器进行器件级建模，非实际流片验证。设计参数基于三星 3D DRAM 规范和 DDR5 标准。Mono3D DRAM 的 scaling 趋势与 3D NAND Flash 一致（已在 400+ 层量产，未来可达 500-1000 层），理论上可支持类似的层数扩展。

涉及论文标题：
- Stratum: System-Hardware Co-Design with Tiered Monolithic 3D-Stackable DRAM for Efficient MoE Serving

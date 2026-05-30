## In-Memory Tiering via WL Staircase Latency (WL 阶梯延迟驱动的内存分层)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
In-Memory Tiering 是 Stratum 利用 Mono3D DRAM 的 WL staircase 结构导致的访问延迟差异，将 1024 层 DRAM 划分为 8 个不同的延迟层级（tier），每个 tier 对应不同的 WL 层区间和不同的 tRCD（RAS-to-CAS delay）值。WL staircase 的物理原理：随着 WL 层向底部延伸（更靠近 staircase 的末端），WL 走线更长，寄生 RC 更大，导致 row activation 延迟线性增加。Stratum 的 8 个 tiers 分别对应 tRCD = [2.29, 3.92, 5.99, 8.50, 11.44, 14.82, 18.63, 22.88] ns，最快 tier（顶层 WL）比最慢 tier（底层 WL）快约 1.6×。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
Tiering 在芯片设计中的实现机制：
1. **Tiering Table**（在 PE 内）：每个 Processing Element 包含一个 16×16b register 的 programmable tiering table，存储 8 个 tier 的末层行地址和对应的 tRCD 值。当 local memory controller 接收到 row access 请求时，将 row address 与 tiering table 存储的边界地址比较，快速确定所属 tier 和对应的 tRCD。
2. **均匀映射策略**：每个 tier 分配相等数量的 DRAM rows（4GB / tier for 32GB chip），通过 configurable tiering table 将物理行地址映射到逻辑 tier。
3. **Data Placement**：SD-Mapper（Algorithm 1）根据 topic-based expert hit rate 将 hot experts 分配到快 tier（低 tRCD）、cold experts 分配到慢 tier（高 tRCD）。非 NMP 数据（xPU 访问的 misc parameters）放在最慢 tier（因需 traversing interposer bottleneck）。
4. **Row-Swap Buffer**：每 PE 内置 8KB row swap buffer，支持 bank 内的 tier-to-tier 数据迁移（expert swap），无需 traversing interposer 接口。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
In-memory tiering 是 Stratum 的原创贡献，无现有商业实现。在更广义的 tiered memory 概念中（如 Intel Optane + DRAM 的两级内存），tier 之间的延迟差异通常达到 3-10×，而 Mono3D DRAM 内的延迟差异仅约 1.6×（因共享相同的存储介质和外围电路）。这种细粒度的内存内部分层需要：(1) 精确的 per-tier timing 参数（通过 Coventor + NeuroSim 仿真获得）；(2) 系统级 data placement 策略（topic-aware mapping）；(3) 硬件支持快速 tier lookup（tiering table 仅占 PE 面积的 0.1%）。该设计的关键洞察是"embracing latency heterogeneity"——不按 worst-case latency 设计（no-tiering），而是利用分层延迟差异提高整体 throughput（1.32-1.45× over no-tiering）。

涉及论文标题：
- Stratum: System-Hardware Co-Design with Tiered Monolithic 3D-Stackable DRAM for Efficient MoE Serving

## 共置 tag 存储（Co-located Tag Storage，ECC 位存 tag 与 Read-Modify-Write）

术语解释
MTE allocation tag 的两种物理组织方式之一：tag 与数据一起存放在专用 meta-data 位中（AmpereOne 用 DDR5 ECC 位，每 16B granule 4 bit），消除 sequestered 方案的 3.03% 软件可见容量开销；另一种是启动时静态划出专用 tag 内存（sequestered），需为全部物理内存预留容量。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
两类开销对比（论文 Fig 3/Fig 4）：(1) 容量开销——sequestered 需保留总物理内存的 3.03%（4 bits/(16B×8+4 bits)），因无法预知哪些区域会被 tag 必须为全部内存预留，直接削减软件可见容量、降低 VM 密度、抬升 CSP TCO；共置方案用 ECC 等 meta-data 位，软件可见容量零开销。(2) 取回/带宽开销——tag 与数据分离时每次 tagged 读写需额外 tag 内存事务（Fig 4 事务数表：分离方案 Read Data+Tag 需 2 个 coherent 事务 vs 共置 1 个；Write Data+Tag 分离需 2 个或 1+1(RMW)），增加带宽与读延迟；共置时读数据即读到 tag（同返），无额外事务。共置的代价：未 tag 的 write-only 事务必须转成 RMW（read-modify-write）以保留已有 tag——这是 AmpereOne 硬件 always-on 开销（B/A 1–6%）的主要来源；RMW 可借 core 比 MCU 更高的带宽部分缓解。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
AmpereOne 的共置实现：MCU 把 4 个 ECC 位/16B granule 用于 tag 存储，设计了 4 种 ECC scheme——A. SECDED-64+8（无 MTE，72 bit codeword=64B 数据+8B ECC，8 codewords/64B cacheline）；B. SECDED-128+4+9（支持 MTE，128B 数据+4B tag+9B ECC，4 codewords/64B，加倍数据粒度腾出闲置 ECC 位、可靠性不削弱）；C. SymbolECC-64+16（无 MTE，Reed-Solomon 每 codeword 纠正 1 symbol）；D. SymbolECC-64-14+2（支持 MTE，从 parity 借 2 bit，100% 检测/100% 有界纠正/99.98% 无界纠正）。MTE 部署用 B/D。mesh 用保留 metadata 位同传 tag、所有 cache line 存储扩展 tag 位且被 ECC 覆盖；core 缓存加宽 tag 与数据共驻。运转流程（一次带 tag 的读）：应用 load → PE cache-lookup（tag 与数据同驻、校验并行）→ miss 则 MCU 从 DRAM 读数据+ECC 位 tag → 经 mesh metadata 位同传 core；一次 untagged full-cacheline store：若共置方案需先读旧 tag（RMW）才能写，否则 tag 丢失。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
同类"用 ECC/meta-data 位存 tag"技术：Sun SPARC ADI、lowRISC Rocket SoC、机密计算与 GPU 内存安全（ISCA'23 的 Implicit Memory Tagging 等）。PCIe 设备无 tag 感知、直接访问内存需保 tag——Ampere 规定软件不把用于 tag 的内存与设备共享，或在这些访问周围保 tag；未来 SMMU tag-aware 化、ARM CHI-E 协议的 tag 管理支持、CXL 3.2 metadata 扩展可为设备路径与 CXL.mem 提供原生 tag 传输。Web 佐证（ARM MTE Performance in Practice，USENIX Sec'26）：Ampere 把 tag 放 ECC 位、cache line 从 64B 扩展为 66B（64B 数据+2B tag）、miss 时单次内存查询取回数据+tag；对比 ARM 参考实现（Pixel 8/9）数据+tag 两次独立读。AmpereOne 参考平台为 Mt. Mitchell（512GB DDR5，SymbolECC）。

涉及论文标题：
- Optimized Memory Tagging on AmpereOne® Processors

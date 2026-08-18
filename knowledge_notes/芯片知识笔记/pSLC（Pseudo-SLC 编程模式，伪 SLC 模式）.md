## pSLC（Pseudo-SLC 编程模式，伪 SLC 模式）

术语解释
- pSLC 是让 TLC 单元只使用最低两个电压态（E、P1）存储 1 bit 的编程模式，以 SLC 级速度（论文实测 114 µs/页）写入，但保留后续经重编程恢复为 TLC 全 8 态容量的能力，与"以 TLC 当 SLC 用、大 ΔVpp 充电"的标准 SLC 模式（96 µs/页、不可恢复）不同。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 标准 SLC 模式（SLC Enable 命令）把 TLC 单元当作 SLC：用大 ΔVpp 一次充到高电压态，速度快但单元被推到高压态、无法再重编程为 TLC 多值态。pSLC 则只把单元编到最低两态之一（E 或 P1），因此：(1) 所需 ISPP 迭代数少、速度接近 SLC（114 µs vs 96 µs）；(2) 单元电压低，不触发 channel pinch-off，为后续长 stride 重编程回 TLC 保留物理可行性。LOONG 用纯编码方式实现 pSLC（无需硬件修改）：把用户页与两个 dummy 页（全 '1'）组成 3 页组，用一次标准 OSP 把 3 页信息编码进前两个电压态（dummy '1' 填充避免了 open block 问题）；第二编程步再把已存页与新两个用户页组成新 3 页组、用一次 OSP 完成到全 8 态的重编程。
- 从芯片设计角度拆解术语：pSLC 是"单元电压态使用方式"层面的设计选择——它把 TLC 单元的 8 态电压窗拆成"第一步只用最低 2 态（快、可靠）、第二步再展开到 8 态（恢复容量）"。可靠性上，pSLC 数据在 1K P/E 与 1 年保留期下误码仅几十/页（ECC 上限 1280 bit/页），且用 OSP 完成（只第一页填用户数据、其余 dummy），规避 open block 问题。相比标准 SLC 模式，pSLC 的关键优势是容量可恢复：SP-GC（用标准 SLC）迁移有效页后需补偿 GC 回收损失容量，而 pSP-GC（用 pSLC）通过重编程直接恢复容量。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现上 pSLC 是固件/FTL 层的编程策略（论文强调无需硬件修改，LOONG 的编程逻辑烧录进控制器、经 mode register 切换，类似 Samsung Turbo Write）。使用场景：(1) GC 优化 pSP-GC——有效页以 pSLC 快速迁移、随后长 stride 重编程恢复容量；(2) 编程优化——两步间被 invalid 的页所在 WL 重编程时用更少电压态；(3) 扩展案例——弹性 SLC 缓存，pSLC 可扩展吸收突发写、耗尽后重编程回 TLC，规避传统固定 SLC 缓存的写悬崖。论文还验证了 QLC 上 pSLC（86 µs）与重编程（4.8 ms）的可行性。

涉及论文标题：
- LOONG: Utilizing Long-Stride Reprogramming to Enhance the Performance of SSDs

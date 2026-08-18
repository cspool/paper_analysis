## Long-Stride Reprogramming（长跨度重编程 / LOONG）

术语解释
- 把两步编程之间的 stride（时间-空间窗口）从 TSP 的 2 WL、[19] 的 8 WL 扩展到整个 block 的重编程机制：第一步以 pSLC 模式顺序编程 block 内全部 WL（低延迟、低电压态），第二步从第一个 WL 起顺序把全部 WL 重编程回 TLC 全 8 态（恢复容量），是 LOONG 论文的核心贡献。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 传统重编程（[19] 等）把 OSP 解耦成快/慢两步，但可靠性（BPD）把两步间隔限制在 8 WL 内，窗口窄导致 GC 中近 1/3 迁移页仍要执行高延迟第二步、且两步间 invalid 页少、无法利用"更少电压态"优化。LOONG 的关键发现：BPD 的严重度由两个因素决定——stride 长度与已编程单元电压，其中 programmed cell voltage（channel pinch-off 效应）是主导项（P3→P7 误码 24576 倍、P1→P7 接近 baseline）。因此把第一步编程电压限制在最低两态（E、P1）即可把 stride 推到整 block。工作流：第一步 pSLC 顺序编程全部 WL（114 µs/页）；第二步长 stride 重编程从第一个 WL 开始（955 µs/页，P1→P7 最坏），统一恢复到 TLC 全 8 态、保留容量。编码上用标准 OSP 实现（3 页组 + dummy 填充），无硬件修改、仅固件（FTL 增加 RBP/SP/RP 指针）。
- 从硬件架构角度拆解术语：长 stride 重编程在 SSD 控制器/固件层实现，核心是"空间-时间解耦"——把高延迟 TLC 编程的开销在更大的时间与空间窗口上推迟分摊，同时保持标准 OSP 作为常态编程策略（LOONG 作为可选的编程策略，只在优化场景启用）。两个案例：(1) GC 优化 pSP-GC——有效页以 pSLC 快速迁移、victim block 擦除后经重编程恢复容量（无 SP-GC 的容量损失/补偿 GC）；(2) 编程优化 TSP-LOONG——两步间 invalid 页大增（较 TSP 增 8.8 倍），含 invalid 页的 WL 重编程只用前 5 态（E→P4，≈710 µs = OSP 的 4/7），读走 MLC 读。扩展案例：弹性 SLC 缓存（pSLC 可扩展 + 重编程恢复）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：固件/FTL 修改（详见 FTL 条目 LOONG 补充）+ 编程引擎内用标准 OSP 编码执行，无需硬件改动；LOONG 编程逻辑烧录进控制器、mode register 切换（类比 Samsung Turbo Write）。可靠性保障：真机验证（FPGA 测试台 + 真实 3D TLC 芯片），1K P/E + 1 年保留期下重编程平均 96 误码/页（ECC 上限 1280 bit/页），且两步编程压低平均 Vth、减少 FN 隧穿泄漏、误码比标准 OSP 更少；长 stride 的扰动效应（disturb）也比 OSP 低（累计操作时间与有效电压差更小）。评估：事件驱动 flash 模拟器 + MSRC/MSPS/FIU/YCSB 工作负载，pSP-GC 平均延迟降 37.5%、TSP-LOONG 平均延迟降 18.1%。

涉及论文标题：
- LOONG: Utilizing Long-Stride Reprogramming to Enhance the Performance of SSDs

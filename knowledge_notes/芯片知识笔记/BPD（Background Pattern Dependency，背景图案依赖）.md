## BPD（Background Pattern Dependency，背景图案依赖）

术语解释
- BPD 指 NAND 单元串中"已编程单元变多 → 串总电阻上升 → 阻碍目标单元电子注入"的物理效应，是 flash 强制顺序编程约束（block 内 WL 必须自下而上单向编程）的根源，也是重编程（reprogram）可靠性的主要威胁。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 单元串（cell string）中同一 bitline 上多个 WL 的单元串联。随着串中更多单元被编程、阈值电压升高，串的总电阻上升（类似闭合电路），已编程单元形成的高电阻会阻碍后续目标单元（下方更未编程的 WL）的电子注入，导致目标单元达不到期望电压态、可靠性与编程成功率下降，即 BPD 引发的可靠性问题。若违反"从 block 底部到顶部 WL 单向顺序编程"的约束（即两步编程之间隔了较多已编程 WL），BPD 会更严重。论文（LOONG）把 BPD 严重度拆解为两个因素并用真机量化：stride 长度（两步之间未编程 WL 数）与已编程单元电压。stride 从 2 增到 12 WL 时误码升 39 倍（指数增长）；而已编程单元电压的影响更剧烈——从 P3 重编程到 P7 时误码较 baseline 飙升 24576 倍，从 P1→P7 则接近 baseline（E→P7 水平）。因此电压因素主导 BPD。
- 从芯片设计角度拆解术语：BPD 是三维 TLC NAND 顺序编程约束的物理根源，直接决定编程/重编程方案能采用多大 stride。论文的量化方法：固定重编程配置（P3→P7）扫描 stride 2–12 WL（图 4a），固定整 block stride 扫描初始态 E/P1/P2/P3→P7（图 4b），测得误码/页分布。结论是整 block 长 stride 之所以可行，全靠把第一步编程电压压到最低两态（P1 及以下）：低电压态使串电阻增量可忽略，channel pinch-off 不触发（见 Channel Pinch-off Effect 条目）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- BPD 无法"实现"，它是物理约束；系统通过编程策略规避它：传统做法是限制重编程 stride ≤8 WL（[19]，MICRO'19 的 8-WL 双层窗口）；LOONG 的做法是第一步限压到 E/P1（pSLC 模式），把可容忍 stride 推到整 block。评估上，论文用真机（FPGA 测试台 + 真实 3D TLC 芯片，232 层/块、16 KB/页）在 1K P/E 周期与 120°C 烘焙 3 小时（≈1 年保留期）下测误码：长 stride 重编程平均 96 误码/页，远低于 ECC 纠错上限（1280 bit/页），且比标准 OSP（最坏 101.8 误码/页）更少——两步编程把平均 Vth 压低，减少 FN 隧穿泄漏。

涉及论文标题：
- LOONG: Utilizing Long-Stride Reprogramming to Enhance the Performance of SSDs

## OSP 与 TSP（One-Shot / Two-Step Programming，一次性/两步编程）

术语解释
- OSP 是 TLC 的主流编程方案：一步内用 ISPP 把单元直接推到 8 个电压态之一（1100 µs/页）；TSP 把编程解耦为两步——第一步粗编程到中间态、第二步细编程（小 ΔVpp）补到目标态，两步之间的时间-空间窗口（stride）被可靠性约束在极窄范围（TSP 2 WL、[19] 8 WL）。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- OSP（One-Shot Programming）：一次编程命令中 ISPP 迭代直到单元达到 8 态中期望的一个，每个态由 LSB/CSB/MSB 三 bit 编码（1-2-4 Gray code 下 E=111、P1=110）。相比 TSP，OSP 在精细电压控制上花的时间少（更少电子注入周期），编程延迟更低；在电荷捕获（CT）NAND（如 Samsung V-NAND）上可靠，是商用高性能 3D SSD 的主流方案。TSP（Two-Step Programming）：第一步粗编程把单元抬到中间态（"1"/"0"，即低/高两档），第二步用更小 ΔVpp 细编程、把额外两个 bit 编程到更高目标态，分布更紧但延迟更高。论文把 OSP 作为默认（与 [19] 对齐），LOONG 完全构建在标准 OSP 之上。
- 从硬件架构角度拆解术语：OSP/TSP 是 SSD 编程引擎（program engine）支持的可选编程模式，两者并存时动态切换 ΔVpp 会给编程引擎引入显著复杂度，因此商用 SSD 通常只实现一种编程机制。LOONG 的挑战之二正是"在既有编程引擎内用 OSP 实现双步编程"——用纯编码（3 页组、dummy 页填充）而非改编程引擎硬件来支持 pSLC + 重编程两种操作。论文实测各编程类型延迟（Table I）：pSLC 114 µs、SLC 96 µs、reprogram 955 µs（P1→P7 最坏）、OSP 1100 µs；reprogram 比 OSP 快（只需 6/7 最大电压），加上第二步读与通道搬运开销，LOONG 相对标准 OSP 总延迟增加不到 2%。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：编程引擎（芯片内）按命令执行 ISPP；编程模式选择由控制器/固件下发。商用产品如 Samsung Turbo Write 通过 mode register 在两种编程逻辑间切换（论文的固件实现参照此法）。使用上，TSP 被用于加速 GC 有效页迁移的第一阶段（快速第一步），但窄窗口导致近 1/3 迁移页仍需执行高延迟第二步；LOONG 用 pSLC（也是"快速第一步"但只编低两态）+ 长 stride 重编程取代 TSP，消除该缺陷（见 Long-Stride Reprogramming 条目）。

涉及论文标题：
- LOONG: Utilizing Long-Stride Reprogramming to Enhance the Performance of SSDs

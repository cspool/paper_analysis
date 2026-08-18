## Bounded Fault（BF，有界故障约束）与 miscorrection 放大

术语解释
约束片上纠错器的 miscorrection 空间范围：任何误纠产生的新错必须落在原故障的有界区域内（不跨区域/符号），从而保证片上纠错只会减少、不会增加系统层看到的符号错误数。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
miscorrection（误纠）指译码器把 y 错误地纠正成另一个合法码字 ĉ ≠ c——错误数反而增多。跨层场景（本文 Motivation）：SEC O-ECC 遇双位错时，约 45% 概率把双位错 miscorrect 成三位错；下游 SEC-DED S-ECC 再把三位错当单位错 miscorrect（约 55% 概率）→ SDC；原始错误率 10^-4 时约每 300 万次访问一次 SDC。Bounded Fault 是 DDR5 引入的规则：O-ECC 的每次纠正必须落在小空间区域（典型 16 位，源自一个 I/O pin/子字线驱动器域）内，保证 miscorrection 不会把错误扩散到系统层的新符号里。矩阵层面的充分条件：H 矩阵中任一区域内列的和 ≠ 该区域外任何列（本文 Fig.2c 的例子：区域内列共享前缀——奇数个错保留前缀仍在区域内，偶数个错前缀相消映射到非数据空间）。Web 佐证：Criss et al.（MEMSYS'20，"Improving Memory Reliability by Bounding DRAM Faults"）定义 DDR5 片上 ECC 码字 128+8，故障分 sub-CL（有界：sub-row/column/DQ pin）与 full-CL（无界：device/bank/row）；HBM3 的有界故障尺寸为 16 位；后续 TVLSI'25 工作把 BF 推广到双错校正 BCH 码（Fault Bounding On-Die BCH Codes）。
从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
BF 作用在片上 O-ECC 解码器的码构造上，是"片上纠错不伤害系统层"的硬件隔离规则。本文 Cerberus 把 BF 落到 16-bit 符号粒度并给出构造：H2（16×288）的每个 16 列有界区域内，后 8 列为前 8 列的 XOR 组合——这样的区域结构同时满足 SEC-DED、BF（区域内列和 = 区域外列的要求成立）与 CRC8（任意 8 连续列线性无关）三重条件。运转流程：bank group 内 Decoder2 读 288b 码字 → 生成 16-bit syndrome → 命中单位错则纠正（新错必在原 16-bit 符号内）→ 多位错时 BF 保证即使 miscorrect 也只改变原符号内容、系统层看到的符号错误数不增加 → Decoder3 的 SSC 仍可把该符号整体纠正。对比 HBM4：其 O-ECC 16-bit SSC 无 BF 约束，miscorrection 可把双符号错扩成三符号错，系统层需 6 符号冗余才能兜住——这正是本文把片上纠错降级为 SEC-DED 的理由。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式（按强度递进）：① 前缀式列布局（DDR5，本文 Fig.2）——区域内列共享前缀；② XOR 半区构造（Cerberus）——区域后 8 列 = 前 8 列 XOR，兼顾 CRC8 与 SSC 兼容；③ BF-BCH 码（TVLSI'25，把 BF 推广到双错校正并解决符号边界 aliasing）。使用时要点：BF 约束 S-ECC 的符号组织（本文指出 DDR5 的 BF 布局迫使 S-ECC 用 Bamboo-ECC 式符号分组），因此跨层码设计时 BF 与系统层符号划分要一起决定（Cerberus 的统一 H2/H^S-ECC 构造即此思路）。场景：所有带片上 ECC 的 DRAM（DDR5/HBM/LPDDR6），尤其当片上纠错能力强（多符号纠错）时 BF 不可或缺；片上只用 SEC-DED 且符号对齐时，BF 是防止 SDC 级联的关键保险。
涉及论文标题：
- Cerberus: Cross-Layer ECC Co-Design for Robust and Efficient Memory Protection

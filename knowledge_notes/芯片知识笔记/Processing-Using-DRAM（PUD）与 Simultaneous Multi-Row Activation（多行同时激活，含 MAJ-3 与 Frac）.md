## Processing-Using-DRAM（PUD）与 Simultaneous Multi-Row Activation（多行同时激活，含 MAJ-3 与 Frac）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- PUD（Processing-Using-DRAM，存内利用式计算）指不修改 DRAM 阵列、直接利用其电学行为（电荷共享、灵敏放大器）完成计算：同一 subarray 内**同时激活多行**（simultaneous multi-row activation，SiMRA）时，多行电容同时在共享位线上放电，位线电压取决于多数行数据，BLSA 放大出的即为"多数表决"结果。MAJ-3（majority-of-three）= 同时激活 3 行做三输入多数表决（MAJ3(a,b,b)=b），基于 MAJ3 可实现 AND/OR/NOT（Ambit 思想）。真实 COTS 芯片经 ACT-PRE-ACT 违规时序可同激活 2/4/8/16/32 行（SiMRA-DRAM，120 颗 DDR4 实测）。Web 来源：Ambit（MICRO 2017）、Simultaneous Many-Row Activation in Off-the-Shelf DRAM Chips（DSN 2024，github.com/CMU-SAFARI/SiMRA-DRAM）、Functionally-Complete Boolean Logic in Real DRAM Chips（HPCA 2024）。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 器件物理链路：ACT 使 wordline 开启 → 行内各单元电容与共享 bitline 电荷共享 → 多行同时开启时 bitline 电压是各电容初值与各存取管导通电阻加权的结果（近似多数/平均）→ BLSA 比较 bitline 与互补 bitline 得到数字化结果。可靠性由存取管电流导通能力的**分布均匀性**决定：任一行导通能力偏离都使电荷共享偏离理想多数函数 → bitline failure。DejaVu 的关键贡献：发现写历史改变该分布——输入行以 OverWrite/SameWrite 初始化后，MAJ-3（16/32 行激活、随机数据、1K 次重复）的 bitline failure rate 比"只写一次"基线平均降 10.7%/32.7%（OverWrite）与 5.8%/30.6%（SameWrite），即"写两次"是零成本的 PUD 可靠性提升手段。芯片级机理假说：写两次使 active region 电荷陷阱占据态更均匀、收窄存取管导通能力分布。配套操作：RowClone（同 subarray 内靠共享 bitline 复制整行，用于反推 subarray 边界）；Frac（Fractional，单元存分数电压使该行对位线电压零贡献，用于处理 MAJ-3 中 N 非 3 倍数的余数行）；MAJ-3 的 32 行实现把三个输入操作数各复制 ⌊32/3⌋=10 份（输入复制冗余提高成功率，SiMRA 报告平均 +30.81%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现流程（DejaVu §8）：RowClone 反推 subarray 边界 → ACT-PRE-ACT 违规时序反推每行的同激活行集 → 每模块随机选 3 个 subarray × 100 组行做 16/32 行 MAJ-3：三输入各复制 ⌊N/3⌋ 份、余数行写 Frac 值 → 同时激活 → 读回与软件多数结果比对，1K 次试验中任一错误即计该 bitline 失败 → 统计 bitline failure rate。使用场景：DRAM 内批量位运算（Ambit/SiMDRAM/SiMRA 的 AND/OR/MAJ）、安全密钥计算、数据库过滤；可靠性瓶颈在于器件变异性（DejaVu 的"写两次"与 SiMRA 的输入复制均可提升成功率）。

PuDGhost 视角（ISCA'26，PuDGhost 论文）：SiMRA/MAJX 的可靠性不仅受器件工艺变异影响，还受"非操作数数据"的电气干扰——同列未激活相邻行（最高 ±10% 偏置）与同 SiMRA 下并发计算列（最高 ±48% 偏置）都会使本列 MAJX 输出出错（Norm. p_o1 偏离 1.0），且 8 行激活是 MAJ3 的常用冗余实现（3 操作数×2 冗余行 + 常量 0/1 行，见隔离行计算行布局条目）。实测中 96 颗 SK Hynix DDR4 芯片均表现出该干扰，说明 PuDGhost 源于 DRAM 电荷共享/SA 采样的普适电学特性（与厂商/代次无关的推断）。
涉及论文标题：
- DejaVu: Why You Should Write to Your DRAM Rows Twice, Carefully

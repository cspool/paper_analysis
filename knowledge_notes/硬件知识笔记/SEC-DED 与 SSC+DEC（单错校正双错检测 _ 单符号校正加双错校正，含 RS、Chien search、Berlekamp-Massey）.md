## SEC-DED 与 SSC+DEC（单错校正双错检测 / 单符号校正加双错校正，含 RS、Chien search、Berlekamp-Massey）

术语解释
两种最常用的内存纠错能力配置：SEC-DED 纠 1 位错检 2 位错（随机软错误场景）；SSC+DEC 纠 1 个符号（多比特簇错）或两个异符号双位错（簇错 + 随机错叠加场景）。RS 码是符号纠错的代表，其解码器用 syndrome 多项式 → BM 迭代 → Chien search 定位的标准流水。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SEC-DED（Single-Error Correction, Double-Error Detection）保证纠正任何单比特错、检测（不纠正）任何双比特错，是最常见主存码（Hamming 及其 Hsiao 优化版，用奇数权重列实现）。SSC+DEC 是本文给系统层定义的复合能力：SSC = 纠正单个符号（本文符号宽 16 位，覆盖 subwordline driver 等外围电路造成的 8–32 位簇错）；DEC = 纠正分布在两个不同符号里的两个位错（如存储错与传输错叠加）；二者 syndrome 空间不重叠（同符号内双位错按符号错处理）。Reed-Solomon 码在 GF(2^m) 上以 2t 个冗余符号纠 t 个符号错，是符号纠错的经典方案（本文 baselines 中 DUO 的 RS(76,64)、Bamboo ECC 的 8-bit 符号均属此类）。逻辑链：DRAM 故障从孤立位错转向空间相关的簇错（SWL/SWD 等共享外围组件一次坏多个相邻位）→ 位级 SEC-DED 不够 → 符号码把空间相关位错折叠成少量符号错，用少量冗余换强纠错。
从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在本文 Decoder3（控制器 S-ECC 硬件）中，SSC+DEC 的实现是两路并行 corrector：SSC corrector 用 Chien search + 修正的 Berlekamp–Massey 过程（RS 标准流水：由 syndrome 构造错误位置/错误值多项式 → BM 迭代求解 → Chien search 逐个符号试根定位错误符号）；DEC corrector 用 block-pair solver（双错校正专用快速逻辑，源自 BCH 双错码实现文献）。两路并行 + 检测单周期，使 error-free 访问不增加 tCL、出错访问单周期完成纠正。能力选择的硬件含义（本文 Device/Link Layer 一节）：片上 O-ECC 若配 8-bit SSC（2 符号冗余），一次 miscorrection 能把双符号错扩成三符号错，系统层需要 6 符号冗余才兜得住——所以 Cerberus 把片上降为 SEC-DED、把 SSC+DEC 集中到系统层，避免片上强纠错的反噬。LPDDR6 的另一极端：S-ECC 同预算只能 SEC-DED（只能纠单位错）或 8-bit SSC（检测能力仅约 86.7%），Cerberus 的 SSC+DEC 在相同 12.5% 预算下两者兼得。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：SEC-DED 的编码器/解码器是纯 XOR 树（本文 Decoder2 仅约 12301 NAND2 等价门、+4 级逻辑深度）；SSC/RS 解码器需要有限域乘加与 Chien search 的扇出结构，面积显著更大（本文 Decoder3 约 124343 NAND2，是全部开销的主导项），但靠"检测单周期 + 纠错并行单周期"控制延迟。使用要点：① 符号宽要对齐物理故障域（本文 16-bit 符号对齐 SWD 故障宽度与 DQ 分组）；② 能力越强 miscorrection 风险越大，需配合 bounded-fault（见下一条目）与层间预算协调；③ 在 32B 小粒度 SDPC 内存上，SSC+DEC 是 32b 冗余预算内能做到的最强组合（本文 Table I 对比：Cerberus 的 Correction=High / Detection=High，优于同预算的 LPDDR6 组合）。
RangeGuard 补充视角（ISCA'26，RS 码符号作用于 RID 元数据而非原始比特）：RangeGuard 在 GPU 的 16-bit parity/256-bit block（6.25%）预算下用 RS 码保护每个值的 RID 符号——8b SSC = 8 个 8-bit RID 数据符号 + 2 个 parity 符号的 RS(10,8)（单符号纠错、最多 32 个损坏数据 bit）；4b DSC = 8 个 4-bit RID 数据符号 + 4 个 parity 符号的 RS(12,8)（双符号纠错、最多 64 个损坏数据 bit；12-symbol 字长满足 4-bit 符号 RS 上限 15）。与 Cerberus 对照：Cerberus 的 16-bit 符号对齐 SWD 物理故障域，RangeGuard 用"每 32-bit 区域内全部值（16/8/4-bit 值的 4/2/1-bit RID）打包成一个 ECC 符号"使 32-bit 对齐 burst 故障表现为单符号错；RangeGuard 的"校正"是把损坏值替换为范围代表值（Bounded Error, BE）而非比特精确恢复，故 Table III 用 BE 替代 CE 报告；解码器用 syndrome + 门电路硬化的改进 Berlekamp-Massey 算法（每纠错 +1 cycle）；Map Tag 可为需要精确恢复的区域切换 SEC-DED（精确但弱）模式。预算效率对比：同 16-bit parity 下 bit 级方案最多纠 8 个翻转 bit，RangeGuard 4b DSC 覆盖 64+ 个翻转数据 bit。
涉及论文标题：
- Cerberus: Cross-Layer ECC Co-Design for Robust and Efficient Memory Protection
- RangeGuard: Efficient, Bounded Approximate Error Correction for Reliable DNNs

## DejaVu（写入历史效应：先前写入的数据影响读干扰、保持性与 PUD 可靠性）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- DejaVu 是 ETH Zurich SAFARI 组（Haocong Luo et al.）在 ISCA 2026 首次实验证实并系统表征的 DRAM 器件现象：**先前写入 DRAM 单元的数据会影响该行后续对读干扰（RowHammer/RowPress）、保持性失效（retention failure）与 PUD 运算可靠性的响应**。三种 victim 行初始化方式对比：Baseline = 只写目标数据一次；OverWrite = 先写反码再覆写目标数据；SameWrite = 相同数据连写两次。在 112 颗 COTS DDR4 芯片（14 条 DIMM，S/H/M 三大厂商，8/16 Gb、A–F 多个 die revision、x8）上：相对 Baseline，OverWrite 使双面 RowHammer ACmin 平均降 2.8%–3.5%（最大 28.1%，即更易翻转），SameWrite 使 ACmin 平均升 3.0%–3.6%（最大 46.1%，即更难翻转）；变化方向在 50/80 °C、0x00/0xFF/0xAA/0x55 全部数据模式下一致。retention：OverWrite（先 0x00 后 0xFF）比 SameWrite 平均多 10.4%（最高 36.7%）bitflip。PUD：OverWrite/SameWrite 初始化使 MAJ-3 的 bitline failure rate 比单写基线降 32.7%/30.6%（32 行激活）。Web 来源：本论文（ISCA 2026，artifact DOI 10.5281/zenodo.19444878，MIT）。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 器件级机理两条假设（论文明确说明均因真实芯片可观测性受限无法完全证实）：(1) **电荷恢复不足**（charge under-restoration）——覆写反码后位线灵敏放大器（BLSA）在标准写恢复时间内无法把电容充/放到目标值；因存取管是 NMOS，写 0xFF 时源端电压升高使 V_GS 下降、导通能力变弱，故 0xFF 更欠恢复（与 Observation 4 一致）；(2) **电荷陷阱占据态改变**（charge trap state change）——写不同值扰动 active region 中陷阱的占据，改变存取管有效阈值电压/导通能力与陷阱辅助漏电。芯片级判别实验：(i) 附加写恢复时间扫描——OverWrite 第二次写需 >400× JEDEC 标准 tWR（>6100ns）才接近 SameWrite 的 ACmin，而 SameWrite 的 ACmin 对附加恢复几乎不变 → 恢复不足假说单独不成立；(ii) 首翻 cache line 空间分布多数模块均匀（仅 M1–M4 偏行尾）→ 行内恢复时间失衡（先写的 cache line 恢复时间更长）也不是唯一机制；(iii) PUD MAJ-3 实验（多行同时激活高度依赖存取管电流分布）中写两次显著降 bitline failure → 支持"写两次使陷阱占据态更均匀、导通能力分布收窄"的假说。芯片设计含义：DRAM 阵列的读干扰/保持性安全裕量不是静态属性，而是依赖每个存储单元的写入历史；任何"victim 行只写一次"的测试或建模都系统性低估（OverWrite 场景）或高估（SameWrite 场景）脆弱性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实验实现：FPGA（Xilinx Alveo U200）+ DRAM Bender 平台直控 DRAM 命令（见 DRAM Bender 条目）；write_row = ACT → 128×WR（覆盖整行 128 cache line）→ PRE，OverWrite/SameWrite 即在 PRE 前追加一次写序列（Listing 1 伪代码）；ACmin 用二分法测量（见 ACmin 条目）。使用方法（论文 §9 建议）：测最坏读干扰阈值时 victim 行一律用 OverWrite 初始化（覆盖 DejaVu 造成的阈值下降）；研究数据模式对读干扰的影响时 victim 行统一用 SameWrite 初始化（避免 for 循环顺序初始化连续行时意外触发 DejaVu 污染结果，见 Listing 2）；retention 测试同理固定初始化协议。系统层应用：DejaVu 使缓解技术阈值需要 guardband——Ramulator 2.0 评估 N_RH −20% 时 PARA 平均性能开销 6.3%。

涉及论文标题：
- DejaVu: Why You Should Write to Your DRAM Rows Twice, Carefully

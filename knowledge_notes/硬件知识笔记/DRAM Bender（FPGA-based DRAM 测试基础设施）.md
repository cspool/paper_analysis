## DRAM Bender（FPGA-based DRAM 测试基础设施）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- CMU-SAFARI 开发的首个开源 DRAM 测试基础设施（IEEE TCAD 2023，arXiv:2211.05838，github.com/CMU-SAFARI/DRAM-Bender）：用 FPGA 直接驱动被测 DRAM 模块（HBM2、DDR4 多种 form factor），主机经 PCIe 下发测试程序，FPGA 以 ±1.5ns 精度按任意顺序、任意时序发送 DRAM 命令，配套 C++/Python 编程接口。六种原型覆盖 Bittware XUSP3S/XUPP3R/XUPVVH、Xilinx Alveo U200/U50 等板卡；VHDL 实现，提供单/双 rank、x4/x8、RDIMM/UDIMM/SODIMM 的预编译 bitstream。Web 来源：github.com/CMU-SAFARI/DRAM-Bender。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 硬件链路（DejaVu Fig.3）：主机 PC（生成测试程序、收集结果）→ PCIe 8-lane → FPGA（执行器，把程序编译为定时精确的 DRAM 命令流）→ DIMM 插座上的被测 DDR4 芯片；heater pads 贴芯片、温控器维持 50/80/95 °C 编程温度。关键能力：(i) 禁用 rank ECC 直接观察电路级 bitflip；(ii) 测试期间停发 auto-refresh（精确时序 + 排除 refresh 干扰）；(iii) 任意扩展 tAggON/tWR 等时序做敏感性扫描；(iv) 发 ACT-PRE-ACT 违规时序实现多行同时激活（PUD 实验）。DejaVu 的三种 victim 初始化（Baseline/OverWrite/SameWrite）就是 FPGA 端 write_row 序列（ACT → 128×WR → PRE）的三种编码。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 使用：按 DRAM Bender README 编译 bitstream（Alveo U200 用 Vivado 2020.2）→ 主机跑测试程序。DejaVu artifact（Zenodo 10.5281/zenodo.19444878）提供 dejavu_ae/run_rd_characterization.sh（读干扰 + retention 表征）与 dejavu_pud/run_pud_reliability.sh（PUD 可靠性），jupyter notebook 出图；温控/加热硬件控制代码因内部基础设施专有未公开（脚本留注释说明接入方式）。用途：RowHammer/RowPress/ColumnDisturb 等读干扰实测、PUD/SiMRA 存内计算实测、retention 与 true/anti-cell 逆向。论文说明未测 DDR5/LPDDR5 正是因为缺少同等精度的对应测试平台。

PuDGhost 视角（ISCA'26）：DRAM Bender 被用于大规模 PuD 可靠性表征——96 颗 SK Hynix DDR4（12 模块，Alveo U200）上执行 2/4/8/16/32 行 APA（ACT-PRE-ACT ≤3ns 间隔）SiMRA、逆向行映射/subarray 边界/even-odd 列、以 50-80°C 温控扫描温度敏感性，并跑两阶段列筛选（8192 样本 screening + 128 样本 execution）与 GEMV/TRNG 应用级评测；PuDGhost 实验代码未单独开源，复用 DRAM Bender/SoftMC/PiDRAM 基础设施。
涉及论文标题：
- PuDGhost: Experimental Analysis of Computation Result Corruption in Processing-using-DRAM Operations on Real DRAM Chips and Implications for Future Systems
- DejaVu: Why You Should Write to Your DRAM Rows Twice, Carefully

## ColumnDisturb（列级读干扰）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- ColumnDisturb 是 2025 年在真实商用 DRAM 芯片上发现的新型**列级**读干扰现象（Yüksel et al., MICRO 2025；获 IEEE Micro Top Picks 2026）。与 RowHammer/RowPress 反复激活或长时间保持"攻击行"不同，ColumnDisturb 反复"锤击"同一**列**——即共享同一条 bitline 的单元。逻辑链：(i) 现代 DRAM 采用开放位线架构，一条列灵敏放大器同时接相邻两个 subarray 的互补 bitline；(ii) 反复激活某一侧的行会持续扰动该列 bitline 的电压偏置；(iii) 共享该 bitline 的所有单元（本 subarray + 相邻 subarray）逐渐泄漏，最终 1→0 翻转。因此一次 hammer 同时影响**三个连续 subarray**、数千行、数百万单元（DDR4 中至多 3072 行）。实测特征（DRAM Bender FPGA 平台，216 颗 DDR4 + 4 颗 HBM2，三家厂商全部受影响）：仅 1→0 翻转；攻击行全 0（被扰动列 GND、受害单元 VDD，全压差）时最严重，最高比全 1 模式多 11.52× 翻转；阈值 N_CD（诱发翻转所需激活次数）随工艺缩进显著下降（到首翻时间最高降 5.06×、平均 2.96×）；标称 64ms 刷新窗口内即可翻转。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 芯片级根源是开放位线组织的共享灵敏放大器结构。攻击流程例子：攻击者经地址映射定位到 bank 内 subarray k 的某一行，反复对其发 ACT → 每次 ACT 同时 hammer (i) subarray k 的全部列 bitline、(ii) 左邻 subarray k-1 的偶列（与 k 共享同一列 SA）、(iii) 右邻 subarray k+1 的奇列 → 三个连续 subarray 中共享该 bitline 的单元在 N_CD 次激活内发生 1→0 翻转。对芯片/系统设计的影响：任何按"行"粒度计数的 RowHammer 防御（TRR、PRAC、PARA、Graphene 等）都无法覆盖"一次破坏三个 subarray 数千行"的破坏面；防御 ColumnDisturb 必须在控制器中暴露（或逆向）subarray 物理映射，或在 DRAM 芯片内部实现计数与刷新。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 研究层面：用 DRAM Bender FPGA 平台（DDR4 用 Xilinx Alveo U200、HBM2 用 Bittware XUPVVH，±1.5ns 命令精度、±0.5°C 控温）实测刻画其阈值、温度/数据模式/时序敏感性。防御层面：本论文 ColumnKeeper 在内存控制器按"subarray × 奇偶列"粒度计数并做轮转预防性刷新（见知识库_硬件架构的 ColumnKeeper-D/P 条目）；Web 来源（ACM DL）：https://dl.acm.org/doi/10.1145/3725843.3756022 。

Sigries 视角（ISCA'26）：ColumnDisturb 的干扰沿列方向传播，受害单元可能离 aggressor 行很远、甚至跨不同 subarray——这种"长程"效应使依赖刷新**邻近** victim 行的 Rowhammer 缓解（包括 Sigries 的 DRFM 式近邻刷新）全部失效。Sigries 论文据此把 ColumnDisturb 列为"当前配置下仅在不实用的服务器设置中被演示过、但未来需严肃对待"的潜在威胁，而非现有攻击面。

涉及论文标题：
- ColumnKeeper: Efficient Solutions to the ColumnDisturb Vulnerability in DRAM-based Systems
- From Lab to Fleet: Building and Deploying a Practical Rowhammer Defense in Cloud SoCs

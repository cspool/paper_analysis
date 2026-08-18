## PuDGhost（PuD 计算结果干扰现象，非操作数数据干扰）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- PuDGhost 是 ISCA 2026 论文（Tokuda 等，UTokyo/ETH Zurich/CISPA/RIKEN）首次在真实 DDR4 芯片上揭示的 Processing-using-DRAM（PuD）计算结果干扰现象：一次 SiMRA 操作中，某列（bitline）的 MAJX 输出会被"非操作数数据"干扰而错误——包括 (1) 未激活的相邻 DRAM 行中存储的数据，(2) 同一 SiMRA 操作下其他并发执行计算的列（即这些列的操作数）。它违反 PuD 的理想假设"每列计算只依赖自身操作数数据"，威胁未来 PuD 系统的可靠性。与 RowHammer/RowPress/ColumnDisturb 等读干扰有本质区别：读干扰由激活的 aggressor 行反复锤击/长时保持引起、表现为受害单元的持久 bitflip；PuDGhost 无需 aggressor 行激活、在单次 SiMRA 的电荷共享与感知期间以瞬态错误形式出现，且刷新/时间间隔无法缓解。Web 来源：arXiv:2606.19119；CISPA 出版物页 https://publications.cispa.de/articles/conference_contribution/PuDGhost_Experimental_Analysis_of_Computation_Result_Corruption_in_Processing-using-DRAM_Operations_on_Real_DRAM_Chips_and_Implications_for_Future_Systems/32781576 ；ISCA26-MCCSys 演讲幻灯片 https://events.safari.ethz.ch/isca26-MCCSys/assets/slides/daichi-isca26-slides.pdf 。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 芯片级机理（论文 §7 假说）：SiMRA 的电荷共享与 SA 感知过程中，未激活相邻行的电荷状态通过电气耦合影响本列电荷共享过程——相邻行存逻辑-0（1）时输出偏置向逻辑-0（1），偏置幅度随相邻行逻辑-1 比例单调增大（Obsv. 4，最高 ±10%）；且存逻辑-1 的激活单元（充电态、放电到 bitline）比逻辑-0 单元更易受相邻行干扰（Obsv. 8）。并发计算列的干扰：Opposite-Parity 列（相邻 subarray 共享 SA）间电气耦合产生单调趋势，Same-Parity 列经共享 SA 电路相互作用产生非单调趋势（Obsv. 13/14），两者叠加成总体 48% 级干扰（Obsv. 10-12）。真实芯片实验流程（DRAM Bender + Xilinx Alveo U200）：96 颗 SK Hynix DDR4（12 模块：TimeTec 3×24 颗 4Gb die A、TeamGroup 7×56 颗 4Gb die M、SK Hynix 2×16 颗 8Gb die A，均 x8）→ 逆向逻辑-物理行映射（RowHammer bitflip）、subarray 边界（RowCopy）、true/anti-cell、even/odd 列 → 每模块 3 subarray × 2/4/8/16/32 行激活 × 128 随机样本：写相邻行指定模式（p_c∈{0,0.25,0.75,1}）→ 写 SiMRA 行随机输入 → APA 序列 SiMRA → 读回算 Norm. p_o1。
- 量化指标：p_o1 = 目标列输出为逻辑-1 的占比；Norm. p_o1 = 某条件下 p_o1 相对基线（相邻行/并发列随机 p_c=0.5）p_o1 的比值，>1 偏逻辑-1、<1 偏逻辑-0，越接近 1 干扰越小。15 条观察覆盖：相邻行干扰高度局域化到物理相邻行（Obsv. 2）、与结构化数据模式无关只与逻辑-1 占比相关（Obsv. 5）、列本地性（每列受自身相邻行单元影响，Obsv. 6/7）、并发列干扰随激活行数增强（Obsv. 11）且强于相邻行干扰（Obsv. 12）、并发列干扰随温度单调变化（Obsv. 15）等。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：PuDGhost 无法靠"更频繁刷新"（0-120ms 等待时间下干扰不变，论文 §8.2 实测）或既有读干扰缓解消除。缓解跨 PuD 计算栈多层：芯片/阵列级方向（isolation transistors、staged activation、reference voltage adjustment，仅讨论）、架构/系统级已实测方案（干扰感知列筛选 CS-1/CS-2 + 隔离行计算行布局，见本库对应条目）。论文提示对 PuD 模拟框架的影响：PIM-SUM 等假设列间错误独立的模型无法刻画 PuDGhost 的应用级质量损失。使用场景：任何依赖 SiMRA 位运算的 PuD 系统（GEMV、TRNG、PUF、数据库过滤）设计时需把 PuDGhost 纳入可靠性预算，尤其是多租户共享 bank 的场景（相邻行数据不可控，存在偏置攻击/信息泄露攻击向量）。

涉及论文标题：
- PuDGhost: Experimental Analysis of Computation Result Corruption in Processing-using-DRAM Operations on Real DRAM Chips and Implications for Future Systems

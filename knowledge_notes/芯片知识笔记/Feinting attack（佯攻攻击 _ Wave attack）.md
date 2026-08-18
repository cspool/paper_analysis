## Feinting attack（佯攻攻击 / Wave attack）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Feinting attack（也称 wave attack）是针对 in-DRAM RowHammer 缓解的模型化最优攻击：攻击者在单个 tREFW 内分多轮，把激活均匀分布在预置的行池上；每当某行计数器超阈值触发缓解（Alert→RFM 或 TRR），攻击者就排除被缓解的行，继续均匀激活剩余行，直到只剩一行存活并承受最大被锤计数（HC）。ProTRR（IEEE S&P 2022，ETH ComSec）形式化证明了该攻击在"按行选择最热门 victim 刷新"的缓解下是最优的，并给出 decoy rows、Feinting-Lite/Medium（DDR5）、Feinting-Postponing RFM/REF、Feinting-Split（利用周期性计数器重置）等变体；Chronus（HPCA 2025）同样以 wave attack 为对手设计 Chronus Back-Off。逻辑链：均匀分布保证任何缓解决策都只能刷新一小部分行，使攻击者总能把扰动集中到最终幸存行上，从而把缓解机制的 NBO 逼到最小安全值，量化防御的 worst-case 安全边界。Web 证据：ProTRR 论文/项目页（comsec.ethz.ch/research/dram/protrr/）、Chronus arXiv:2502.12650、PVAC arXiv:2604.20576（引用 [57] ProTRR、[98] wave attack）。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 PVAC 的芯片设计分析中，feinting attack 用于推导 victim 计数与 aggressor 计数在相同安全约束下各自的 NBO（§VI，参照 [91] 的方法论但按 victim 侧 HC 重述）。运转流程（PVAC，BR=2）：Setup phase——攻击者以 stride=5 激活（避免 victim 的 BR 重叠），把预置 victim 池 R1 的每个 victim 的对应 aggressor 行各激活 NBO-1 次，得 HC_setup=NBO-1；Online phase——共 NR 轮，每轮对剩余行各激活一次，被 RFM 刷新的行（每 Alert 触发 NMit×4 行）被排除，直到 R_n=1；最终轮还可追加 ABO_ACT+ABO_Delay 次激活与 BR 次 RFM 前扰动，得 HC_online=NR+ABO_Delay+ABO_ACT+BR。对 PRAC（aggressor 计数）的对应建模：Setup 的 2×BR 个最终 aggressor 行的扰动在 victim 处累加，HC_setup=2×BR×(NBO-1)，且 NBO 必须压到 HC/(2×BR) 量级。结果对比（HC=128、NMit=1/2/4）：PVAC NBO=85/102/108 vs PRAC 3/15/19 vs Chronus 31——victim 计数因不按 BR 分摊而获得大得多的安全 NBO，这正是 PVAC 芯片级参数（NBO、优先级队列深度、RFM 频率）的推导来源。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：攻击模式本身是地址序列——攻击者用支持行粒度寻址的普通访存指令（同一 bank/row 的激活）即可构造，无需芯片内部访问；防御方在模拟器（Ramulator 2.0）中对攻击模式建模：把 ABO 协议参数（ABO_ACT=3、ABO_Delay=NMit、每 Alert NMit 个 RFM、每个 RFM 350ns、tRC=48ns）与行池递推式（R_n=R_(n-1)-NMit×floor(R_(n-1)/(ABO_ACT+ABO_Delay))，PRAC 版为 R_n=R_(n-1)-NMit×(R_(n-1)-BR)/(ABO_ACT+ABO_Delay)）代入求 NR，再用 HC 公式反解 NBO。意义：给出缓解机制的 worst-case 带宽/性能上界（附录 XI 用 NMit×tABO_Recovery/(NMit×tABO_Recovery+NBO×tRC) 估算最大 RFM 带宽占比，PVAC-4 在 HC=256 时为 11.0% vs PRAC-4 的 34.1%）；也揭示固定 NMit 的 PRAC 类方案在单行流下的保守性（Chronus 自适应 NMit 在该分析中优于 PVAC-4，但实证多负载下 PVAC 更优）。

涉及论文标题：
- PVAC: A RowHammer Mitigation Architecture Exploiting Per-victim-row Counting

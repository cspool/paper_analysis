## DLDO（Digital Low-Dropout Regulator，数字低压差稳压器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- DLDO 是片上集成的低压差线性稳压器（LDO）的数字实现：把较高的输入电压（如 1.15V）降到目标输出电压（如 0.8–1.1V），核心是 PMOS pull-up 阵列作为可调导通器件，由数字控制环（比较器 + 计数器/移位寄存器 + 译码器）按负载反馈调节导通 PMOS 数量（对应数字电压级数）以稳定输出。与模拟 LDO 相比，DLDO 控制逻辑全数字、可综合、易迁移工艺，缺点是输出纹波与瞬态响应受限于开关粒度。它常用于细粒度 DVFS：为每个电压-频率域提供独立、可快速切换的本地电源，是"空间 DVFS/每域独立调压"的关键使能器件。
- PowerWeave（ISCA'26）用 DLDO 作为空间 DVFS 每域的片上供电方案：每个频率域配一个专用 on-die DLDO。论文建模其面积与功耗开销：输入 1.15V、输出 0.8–1.1V、最大步长 1% Vout（≈11mV）、最小分辨率 128 级（保守取 256 级）；功率部分（PMOS 阵列）面积与需供应的峰值电流成正比，GPU 峰值电流假设恒定、与 DVFS 粒度无关。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 芯片设计中的运转流程：DVFS 域数 N 增加 → 每域复制一套 DLDO → 由于功率器件（PMOS 阵列）总面积随总电流恒定（N 域分摊同一总电流），增量开销只来自复制控制逻辑：ΔA_reg(N) = A_DLDO,ctrl^5nm × (N−1)，其中 A_DLDO,ctrl^5nm = A_DLDO,ctrl^7nm / S_7nm→5nm（IRDS 数字面积缩放因子）。论文用 OpenFASoC DLDO generator 生成参数化 regulator RTL → 提取控制逻辑 → ASAP7 7nm PDK 全 RTL-to-GDSII 流程（mflowgen 驱动）→ 提取 post-layout 面积（Table IV：每额外域 0.0023 mm²，占 1600 mm² die 的 0.00009%）。功耗：DLDO controller 7nm 综合（workload-driven activity annotation）78 µW/regulator，148 域合计 ≈11.5 mW，相对 B200 的 1000W TDP 可忽略。
- 设计权衡例子：per-GPC（粗）到 per-SM（148 域，最细）粒度扫描（Fig. 13），regulator 项随 N 线性增长但绝对量很小——per-SM 时三项总和仍 <0.5% die 面积，其中 DLDO 占比最小，说明稳压器不是空间 DVFS 的成本主导项。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：业界将 DLDO 与片上集成电压调节器（IVR）结合用于 per-core/per-block DVFS——Intel FIVR（4th gen Core）、22nm 图形执行核心的数字控制全集成稳压器 [25]、14nm 三栅图形处理器 fine-grain DVFS + 执行单元 turbo + retentive sleep [33]、AMD Zen/Zeppelin 的集成电源管理。论文的落地方式：OpenFASoC（https://github.com/idea-fasoc/OpenFASOC）开源自动模拟块生成框架生成可综合 DLDO RTL，ASAP7 7nm PDK（https://github.com/The-OpenROAD-Project/ASAP7）综合布局布线，mflowgen（https://github.com/mflowgen/mflowgen）搭 RTL-to-GDSII 流程，IRDS（https://irds.ieee.org/）技术缩放。用于评估"每个 SM 独立电压域"的硬件代价，结论：per-SM 148 域 regulator 面积增量仅 0.0023 mm²/域。

涉及论文标题：
- PowerWeave: Unlocking Energy-Efficient ML on GPUs with OS-Level Spatial Power Management

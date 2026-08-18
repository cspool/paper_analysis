## A Streaming Architecture for Quantum Error Syndrome Compression at 4 Kelvin

- 属于硬件架构的实现是什么？实验比较什么？
  - 实现为 IcePack：面向 4 K 层的 tiled、参数化 SFQ 流式微架构，压缩后 syndrome 经单根电缆上行到 300 K 解码器；每 tile 由三部分组成：(1) PPU 预处理单元——syndrome 位流按 block unit（BU）分块，每 BU 用 1 个 DRO 单元（功能同 D 触发器）检测全零块并跳过（all-zero block filtering），非零块数据存于 PTL 序列存储器，经共享 priority selector（按升序选块）+ NDRO 单元（掩码未选中块）+ merger tree 串行送出；(2) PU 处理单元——时钟 RSFQ 门实现的 7 级流水线（每级处理位流中不同位置），含 SCU 空间聚类单元（移位寄存器/PTL 行缓冲构成 5-ancilla 滑窗，组合逻辑查真值表输出 2-bit opcode + 1-bit valid）与 TCU 时间聚类单元（将 SCU 输出 (V_in, OP_in) 与上一轮预测流 P_in 对比更新 (V_o, OP_o)、生成下一轮预测 P_o；共享计数器采运行 index 写入队列；PTL 环形延迟结构存整轮预测）；(3) ENC 编码单元——减法器算 gap、硬连线移位取商/余数（Rice-Golomb m=2^k）、计数器做 binary→unary 转换，用 xSFQ 无时钟门实现。存储采用 delay-line memory (DLM)：以被动传输线（PTL）为延迟介质，同步控制器 + 按需插入 DRO 单元分段防抖动，10 GHz 下每 PTL 可存至多 41 bit，JJ/bit 比移位寄存器方案少 40×。
  - 实验比较：vs 全并行 SFQ 解码器（NISQ+、QECOOL、Clique——Clique ≥96 JJ/ancilla、处理 <0.3 ns 但假设 1 μs 串行化）与"稀疏表示流式 SFQ baseline"（从 IcePack 裁掉 SCU/TCU/ENC/预测存储，仅留 BU + Priority Selector）；指标：JJ 数/ancilla（PU/PPU/ENC 分项）、PTL 面积、功耗与热负载（含电缆）、99 分位处理延迟、300 K 解压延迟、DLM 原型流片实测。
- 硬件平台是什么，配置是什么。
  - 逻辑族：时钟 RSFQ 门（PU，7 级流水）+ xSFQ 无时钟门（ENC，sub-GHz 匹配索引传输）；假设 10 GHz SFQ 运行（保守，文献与本文实测支持 16–33 GHz）；上行电缆 1 Gb/s 不锈钢同轴（1 mW/Gb/s + 10.5 mW 外设功耗）；表面码 d=21。原型：2 mm Nb 延迟线存储环，MITLL SFQ5ee 工艺流片，实测 33 GHz 循环存储 2 个 SFQ 脉冲（模拟电压读出幅值与脉冲数成正比）。
- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  - PyLSE（UCSBarchlab，https://github.com/UCSBarchlab/PyLSE）做门级功能仿真：随机输入、数千测量轮，以自研 IcePack emulator（syndrome 压缩评估用，Section VI-A）为黄金参考；论文未说明修改 PyLSE 本体。300 K 解压器用 Synopsys DC + Nangate 45nm 库综合。
- 模拟器模拟什么的性能，修改了什么。
  - PyLSE 对 SFQ 网表做脉冲级（pulse-transfer level）功能仿真，逐 bit 验证 PPU/PU/ENC 压缩逻辑与黄金参考一致；未报告被修改。面积/功耗按文献 SFQ 单元与 PTL 时序参数分析（JJ 0.2 aJ/次开关、偏置 +50% 开销、20% 单元/1% PTL 时序变异）；队列占用用 Stim 生成的 syndrome index 分布驱动 10 万周期仿真取 99 分位延迟。
- 开源情况。基于开源文档和论文，使用例子解释模拟器如何使用？作用是什么？至少具体到模拟器模拟性能的原理和模拟器输入到性能输出的全过程。
  - SFQ 硬件网表（PyLSE 模型）是否开源论文未明确说明；算法评估 artifact 开源（Zenodo https://doi.org/10.5281/zenodo.19446086，CC BY 4.0）。PyLSE 为开源 Python 嵌入 DSL（UCSB archlab）：以 Wire/Element/Circuit 类描述超导电子单元、脉冲编码，内建仿真框架跑波形（inspect() 输出），可用 UPPAAL 做模型校验。
  - 全过程：输入 = 行主序 syndrome 位流（每测量轮每 ancilla 1 bit）+ 4 K 时钟域控制；PPU 按块 DRO 过滤全零块 → PU 7 级流水滑窗模式匹配（SCU 真值表 (V, OP) → TCU 预测对比丢/补 index）→ ENC 移位变长编码 → 队列串行写出经单根 1 Gb/s 电缆上行；输出 = JJ 数分项（Table III：PU 1502 恒定、PPU/ENC 随错误率/块数变化，平均 4 JJ/ancilla、范围 3.2–13.6）、PTL 面积（SFQ5ee Nb stripline 3000 μm²/ancilla、SC2 MoN stripline 187 μm²/ancilla，最多 50 万 qubit/cm²）、功耗 10–42 nW/ancilla（JJ 全开最坏情形）、热负载/延迟 Pareto（vs 数字读出热负载 -11×、延迟 -10×）、99 分位处理延迟（block≤128 时 ≤17 ns，占 500 ns 目标 <4%）、300 K 解压 2.5 ns。
